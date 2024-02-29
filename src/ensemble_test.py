import torch.nn as nn
from test import TrainedFolder
import os.path as osp
from datetime import datetime
import glob
import yaml
import torch
from utils.utils_functions import get_device, get_split
import os
from test import init_model_test
import argparse
import utils.nmr_dataset as nmr_dataset
import pandas as pd


class EnsembleTrainedModel(nn.Module):
    """
    Ensemble of trained models for test only. Not able to be trained.
    """
    def __init__(self, model_list, pooling='mean'):
        super().__init__()
        self.model_list = model_list
        self.pooling = pooling

    def forward(self, data):
        out_list = [model(data)['atom_prop'] for model in self.model_list]
        return torch.mean(torch.stack(out_list), dim=0)


class EmsembleTrainedFolder(TrainedFolder):
    """
    Class to evaluate ensemble models. 
    """
    def __init__(self, 
                 folder_name_list,
                 test_dir_prefix='test', 
                 dataset_args=None,
                 dataset_class=None,
                 split=None,
                 all_test=0,
                 labeled_data=False,
                 save_csv=False,
                 ignore_train=True, 
                 ignore_val=False):
        self.test_dir_prefix = test_dir_prefix
        self.dataset_args = dataset_args
        self.dataset_class = dataset_class if dataset_class is not None else 'NMRDatasetFromProcessed'
        self.split_file_path = split
        self.all_test = all_test
        assert self.all_test > 0 or self.split_file_path is not None
        self.labeled_data = labeled_data
        self.save_csv = save_csv
        self.folder_name_list = folder_name_list
        super().__init__(ignore_train=ignore_train, ignore_val=ignore_val)
    
    def get_test_dir(self):
        return self.test_dir_prefix + '_' + datetime.now().strftime('%m%d_%H%M%S')

    def get_dataset(self):
        dataset_class = getattr(nmr_dataset, self.dataset_class)
        self.data_provider = dataset_class(**self.dataset_args)
        self.splits = get_split(self.split_file_path) if self.split_file_path else get_split(all_test=self.all_test)

    def get_args(self):
        # read all the config files in the folder_name_list
        args_list = []
        for folder_name in self.folder_name_list:
            config_name = glob.glob(osp.join(folder_name, 'config.yaml'))[0]
            with open(config_name, 'r') as yaml_file:
                args_list.append(yaml.safe_load(yaml_file))
        # all the args should be the same for ensemble models
        assert all(args == args_list[0] for args in args_list)  # TODO: discard some keys that are not necessary to be the same
        self.config_name = glob.glob(osp.join(self.folder_name_list[0], 'config.yaml'))[0]
        return args_list[0]

    def load_model(self):
        self.ensemble_model_list = []
        for folder_name in self.folder_name_list:
            model_data = torch.load(os.path.join(folder_name, 'best_model.pt'), map_location=get_device())
            model = init_model_test(self.args, model_data)
            self.ensemble_model_list.append(model)
        ens_model = EnsembleTrainedModel(self.ensemble_model_list)
        return ens_model

    def test_step(self, data_loader, result_file):
        if self.labeled_data:
            return super().test_step(data_loader, result_file)
        else:
            return self.test_step_unlabeled(data_loader, result_file, save_csv=self.save_csv)

    def test_step_unlabeled(self, data_loader, result_file, save_csv=False):
        """test step for unlabled data, the loss is not calculated. Prediciton and atom index is saved."""
        self.model.eval()
        prop_pred = []
        atom_index = []
        with torch.set_grad_enabled(False):
            for val_data in data_loader:
                val_data = val_data.to(get_device())
                this_prop_pred = self.model(val_data)
                if self.args['mask_atom']:
                    mask = val_data.mask.bool()
                    this_prop_pred = this_prop_pred[mask, :]
                    this_atom_index = val_data.mask.nonzero(as_tuple=True)[0]
                else:
                    this_atom_index = torch.arange(val_data.N)
                prop_pred.append(this_prop_pred)
                atom_index.append(this_atom_index)
        result = {'PROP_PRED': torch.cat(prop_pred, dim=0), 'ATOM_INDEX': torch.cat(atom_index, dim=0)}
        torch.save(result, result_file)
        if save_csv:
            result_csv = result_file.replace('.pt', '.csv')
            result_df = pd.DataFrame(result['PROP_PRED'].detach().cpu().numpy(), columns=['PROP_PRED'])
            result_df['ATOM_INDEX'] = result['ATOM_INDEX'].detach().cpu().numpy()
            result_df.to_csv(result_csv, index=False)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ens_config', type=str, help="Ensemble test config file")
    _args = parser.parse_args()
    with open(_args.ens_config, 'r') as yaml_file:
        ens_configs = yaml.safe_load(yaml_file)
    tester = EmsembleTrainedFolder(**ens_configs)
    tester.run_test()
