import argparse
import logging
import math
import os.path as osp
import shutil
import time
from collections import OrderedDict
from copy import copy
import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from Networks.PhysDimeNet import CSTShiftEmb, CSTShiftOut, sPhysNet
import utils.nmr_dataset as nmr_dataset
from utils.LossFn import LossFn
from utils.Optimizers import EmaAmsGrad
from utils.tags import tags
from utils.utils_functions import floating_type, get_lr, get_n_params, \
    remove_handler, option_solver, get_device, \
    validate_index, get_folder_name, solv_num_workers, get_train_atomic_idx, get_ext_atom_feat_mean_std, get_git_revision_short_hash, get_split
import yaml


class Trainer:
    def __init__(self, 
                 config_args, 
                 chk=None, 
                 num_workers=None, 
                 toy_split=0,
                 folder_suffix="",
                 **kwargs):
        self.config_args = config_args

        self._run_directory = None
        self._sum_writer = None
        self._logger = None
        self._loss_csv = None
        self._loss_df = None
        self.chk = chk
        self.num_workers = num_workers
        self.toy_split = toy_split
        self.folder_suffix = folder_suffix

    def train(self, data_provider=None, ignore_valid=False):
        self.log('git hash: {}'.format(get_git_revision_short_hash()))
        n_cpu_avail, n_cpu, num_workers = solv_num_workers()
        self.log(f"Number of total CPU: {n_cpu}")
        self.log(f"Number of available CPU: {n_cpu_avail}")
        self.log(f"Number of workers used: {num_workers}")
        if self.num_workers is not None:
            num_workers = self.num_workers
            self.log(f"Number of workers overwritten in config: {num_workers}")

        # ------------------- dataset setting ---------------------- #
        config_dict = self.config_args

        data_class = getattr(nmr_dataset, config_dict["data"]["dataset_class"]) if "dataset_class" in config_dict["data"] else nmr_dataset.NMRDatasetFromProcessed
        data_provider = data_class(**config_dict['data']['dataset_args'])
        self.log("used dataset: {}".format(data_provider.processed_file_names))
        split_file_path = config_dict['data']['split'] if 'split' in config_dict['data'] else config_dict['data']['dataset_args']["root"] + 'processed/split.pt'
        splits = get_split(split_file_path, toy_split=self.toy_split)
        train_index, val_index, test_index = splits['train_index'], splits['val_index'], splits['test_index']
        train_size, val_size, test_size = validate_index(train_index, val_index, test_index)
        self.log(f"train/validation/test size: {train_size}/{val_size}/{test_size}")

        train_data_loader = DataLoader(
            data_provider[torch.as_tensor(train_index)], batch_size=config_dict["batch_size"],
            pin_memory=True, shuffle=True, num_workers=num_workers)

        val_data_loader = DataLoader(
            data_provider[torch.as_tensor(val_index)], batch_size=config_dict["valid_batch_size"],
            pin_memory=True, shuffle=False, num_workers=num_workers)

        loss_fn = LossFn(**config_dict)

        # ------------------- Setting up model and optimizer ------------------ #
        # Normalization of PhysNet atom-wise prediction
        mean_atom = []
        std_atom = []
        train_atomic_idx = get_train_atomic_idx(data_provider, train_index, config_dict["mask_atom"])

        for name in config_dict["target_names"]:
            this_atom_prop: torch.Tensor = getattr(data_provider.data, name)[train_atomic_idx]
            mean_atom.append(torch.mean(this_atom_prop.double()).item())
            std_atom.append(torch.std(this_atom_prop.double()).item())
        mean_atom = torch.as_tensor(mean_atom)
        std_atom = torch.as_tensor(std_atom)
        config_dict['mean_atom'] = mean_atom
        config_dict['std_atom'] = std_atom

        if 'ext_atom_method' in config_dict:
            # infer the dimension of external atom feature
            config_dict['ext_atom_feat_mean_std'] = get_ext_atom_feat_mean_std(data_provider, train_index,
                                                                               config_dict['ext_atom_features'],
                                                                               config_dict['ext_atom_dim'],
                                                                               config_dict['ext_atom_features_norm'])

        if config_dict.get('ext_atom_method') == 'emb':
            net = CSTShiftEmb(**config_dict)
        elif config_dict.get('ext_atom_method') == 'out':
            net = CSTShiftOut(**config_dict)
        else:
            net = sPhysNet(**config_dict)
        net = net.to(get_device())
        net = net.type(floating_type)

        shadow_dict = None

        # optimizers
        optimizer = EmaAmsGrad(net, lr=config_dict["learning_rate"],
                               shadow_dict=shadow_dict, params=net.parameters())

        # schedulers
        scheduler_kw_args = option_solver(config_dict["scheduler"], type_conversion=True)
        scheduler_base = config_dict["scheduler"].split("[")[0]
        config_dict["scheduler_base"] = scheduler_base
        if scheduler_base == "StepLR":
            if "decay_epochs" in scheduler_kw_args.keys():
                step_per_epoch = 1. * train_size / config_dict["batch_size"]
                decay_steps = math.ceil(scheduler_kw_args["decay_epochs"] * step_per_epoch)
            else:
                decay_steps = config_dict["decay_steps"]
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_steps, gamma=0.1)
        elif scheduler_base == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kw_args)
        elif scheduler_base == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **scheduler_kw_args)
        else:
            raise ValueError('Unrecognized scheduler: {}'.format(config_dict["scheduler"]))

        # --------------------- Printing data ---------------------- #
        if get_device().type == 'cuda':
            self.log('Hello from device : ' + torch.cuda.get_device_name(get_device()))
            self.log("Cuda mem allocated: {:.2f} MB".format(torch.cuda.memory_allocated(get_device()) * 1e-6))

        n_parm, _ = get_n_params(net, None, False)
        self.log('model params: {}'.format(n_parm))
        n_parm, _ = get_n_params(net, None, True)
        self.log('trainable params: {}'.format(n_parm))

        # ---------------------- Training ----------------------- #
        self.log('start training...')
        t0 = time.time()

        shadow_net = optimizer.shadow_model
        val_res = val_step_new(shadow_net, val_data_loader, loss_fn)

        valid_info_dict = OrderedDict(
            {"epoch": 0, "train_loss": -1, "valid_loss": val_res["loss"], "delta_time": time.time() - t0})
        for key in val_res.keys():
            if key != "loss" and not isinstance(val_res[key], torch.Tensor):
                valid_info_dict[key] = val_res[key]
                valid_info_dict.move_to_end(key)
        self.update_df(valid_info_dict)

        # use np.inf instead of val_res["loss"] for proper transfer learning behaviour
        best_loss = np.inf

        if self.chk:
            last_epoch = pd.read_csv(osp.join(self.run_directory, "loss_data.csv"), header="infer").iloc[-2]["epoch"]
            last_epoch = int(last_epoch.item())
            self.log('Init lr: {}'.format(get_lr(optimizer)))
        else:
            last_epoch = 0

        early_stop_count = 0
        step = 0

        self.log("Setup complete, training starts...")

        for epoch in range(last_epoch, last_epoch + config_dict["num_epochs"]):
            # Early stop when learning rate is too low
            this_lr = get_lr(optimizer)
            if config_dict["stop_low_lr"] and this_lr < 3 * getattr(scheduler, "eps", 1e-9):
                self.log('early stop because of low LR at epoch {}.'.format(epoch))
                break

            train_loss = 0.
            for batch_num, data in enumerate(train_data_loader):
                data = data.to(get_device())
                this_size = data.N.shape[0]

                train_loss += self.train_step(net, _optimizer=optimizer, data_batch=data, loss_fn=loss_fn,
                                              scheduler=scheduler, config_dict=config_dict) * this_size / train_size
                step += 1

            # ---------------------- Post training steps: validation, save model ---------------- #
            self.log('epoch {} ended, learning rate: {} '.format(epoch, this_lr))
            shadow_net = optimizer.shadow_model
            val_res = val_step_new(shadow_net, val_data_loader, loss_fn)
            if config_dict["scheduler_base"] in tags.step_per_epoch:
                scheduler.step(metrics=val_res["loss"])

            valid_info_dict = {"epoch": epoch, "train_loss": train_loss, "valid_loss": val_res["loss"],
                                "delta_time": time.time() - t0}
            for key in val_res.keys():
                if key != "loss" and not isinstance(val_res[key], torch.Tensor):
                    valid_info_dict[key] = val_res[key]
            self.update_df(valid_info_dict)
            self.summarize_epoch(**valid_info_dict)
            t0 = time.time()

            if ignore_valid or (val_res['loss'] < best_loss):
                early_stop_count = 0
                best_loss = val_res['loss']
                torch.save(shadow_net.state_dict(), osp.join(self.run_directory, 'best_model.pt'))
                torch.save(net.state_dict(), osp.join(self.run_directory, 'training_model.pt'))
                torch.save(optimizer.state_dict(), osp.join(self.run_directory, 'best_model_optimizer.pt'))
                torch.save(scheduler.state_dict(), osp.join(self.run_directory, "best_model_scheduler.pt"))
            else:
                early_stop_count += 1
                if early_stop_count == config_dict.get("early_stop", -1):
                    self.log('early stop at epoch {}.'.format(epoch))
                    break

        remove_handler(self.logger)

    def train_step(self, model, _optimizer, data_batch, loss_fn, scheduler, config_dict):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.autograd.set_detect_anomaly(False):
            model.train()
            _optimizer.zero_grad()
            model_out = model(data_batch)
            loss = loss_fn(model_out, data_batch)
            loss.backward()

        clip_max_norm = config_dict.get("clip_grad_max_norm", 1000.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        _optimizer.step()
        if config_dict["scheduler_base"] in tags.step_per_step:
            scheduler.step()

        result_loss = loss.data.cpu().item()
        return result_loss

    def update_df(self, info_dict: dict):
        new_df = pd.DataFrame(info_dict, index=[0])
        updated = pd.concat([self.loss_df, new_df])
        updated.to_csv(self.loss_csv, index=False)
        self.set_loss_df(updated)

    def log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def set_loss_df(self, df: pd.DataFrame):
        if self._loss_df is None:
            # init
            __ = self.loss_df
        self._loss_df = df

    def summarize_epoch(self, epoch, delta_time, **scalars):
        for key in scalars:
            self.sum_writer.add_scalar(key, scalars[key], global_step=epoch, walltime=delta_time)

    @property
    def sum_writer(self):
        if self._sum_writer is None:
            self._sum_writer = SummaryWriter(self.run_directory)
        return self._sum_writer

    @property
    def loss_df(self):
        if self._loss_df is None:
            if osp.exists(self.loss_csv):
                # load pretrained folder csv
                loss_df = pd.read_csv(self.loss_csv)
            else:
                loss_df = pd.DataFrame()
            self._loss_df = loss_df
        return self._loss_df

    @property
    def loss_csv(self):
        if self._loss_csv is None:
            self._loss_csv = osp.join(self.run_directory, "loss_data.csv")
        return self._loss_csv

    @property
    def logger(self):
        if self._logger is None:
            # --------------------- Logger setup ---------------------------- #
            # first we want to remove previous logger step up by other programs
            # we sometimes encounter issues and the logger file doesn't show up
            log_tmp = logging.getLogger()
            remove_handler(log_tmp)
            logging.basicConfig(filename=osp.join(self.run_directory, self.config_args["log_file_name"]),
                                format='%(asctime)s %(message)s', filemode='w')
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            self._logger = logger
        return self._logger

    @property
    def run_directory(self):
        if self._run_directory is None:
            # ----------------- set up run directory -------------------- #
            if self.chk is None:
                run_directory = get_folder_name(suffix=self.folder_suffix)
                shutil.copyfile(self.config_args["config_name"], osp.join(run_directory, f"config.yaml"))
            else:
                run_directory = self.chk
            self._run_directory = run_directory
        return self._run_directory


def val_step_new(model, _data_loader, loss_fn: LossFn, mol_lvl_detail=False, lightweight=True, config_dict=None):
    """
    Kina messy, but it works, for now.
    :param model:
    :param _data_loader:
    :param loss_fn:
    :param mol_lvl_detail:
    :param lightweight: delete atom embedding otherwise result becomes too large
    :param config_dict:
    :return:
    """
    model.eval()
    valid_size = 0
    loss = 0.
    loss_detail = None
    with torch.set_grad_enabled(False):
        for val_data in _data_loader:
            val_data = val_data.to(get_device())
            model_out = model(val_data)
            aggr_loss, new_loss_detail = loss_fn(model_out, val_data, loss_detail=True, mol_lvl_detail=mol_lvl_detail)
            # n_units is the batch size when predicting mol props but number of atoms when predicting atom props.
            n_units = new_loss_detail["n_units"]
            loss += aggr_loss.item() * n_units
            if loss_detail is None:
                loss_detail = init_loss_detail(new_loss_detail)
            loss_detail = update_loss_detail_in_step(loss_detail, new_loss_detail, n_units, valid_size)
            valid_size += n_units
    loss_detail = update_loss_detail_after_step(loss_detail, valid_size, loss)
    return loss_detail


def init_loss_detail(new_loss_detail):
    """
    initialize loss detail dictionary.
    """
    loss_detail = copy(new_loss_detail)
    for key in new_loss_detail.keys():
        if tags.val_avg(key):
            loss_detail[key] = 0.
        elif tags.val_concat(key):
            loss_detail[key] = []
        else:
            # we do not want temp information being stored in the final csv file
            del loss_detail[key]
    return loss_detail


def update_loss_detail_in_step(loss_detail, new_loss_detail, n_units, valid_size):
    """
    update loss detail dictionary during the validation step. The results are not averaged yet.
    """
    for key in loss_detail:
        if tags.val_avg(key):
            loss_detail[key] += new_loss_detail[key] * n_units
        elif tags.val_concat(key):
            loss_detail[key].append(new_loss_detail[key])
    return loss_detail


def update_loss_detail_after_step(loss_detail, total_valid_size, loss):
    """
    update loss detail dictionary after the validation steps. The results are averaged here.
    """
    loss_detail["n_units"] = total_valid_size
    loss /= total_valid_size
    for key in list(loss_detail.keys()):
        if tags.val_avg(key):
            loss_detail[key] /= total_valid_size
        elif tags.val_concat(key):
            loss_detail[key] = torch.cat(loss_detail[key], dim=0)
        # add RMSE
        if key.startswith("MSE_"):
            loss_detail[f"RMSE_{key.split('MSE_')[-1]}"] = math.sqrt(loss_detail[key])
    loss_detail["loss"] = loss
    return loss_detail


def train(config_dict=None, data_provider=None, ignore_valid=False):
    trainer = Trainer(config_dict, **config_dict)  # TODO better way to pass arguments
    trainer.train(data_provider, ignore_valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='configs/nmr_test.yaml')
    config_args = parser.parse_args()
    with open(config_args.config_name, 'r') as yaml_file:
        args = yaml.safe_load(yaml_file)
    args['config_name'] = config_args.config_name
    train(args)
