from copy import deepcopy
from typing import Union, List
import torch
from utils.tags import tags


class LossFn:
    def __init__(self,
                 target_names=None, 
                 loss_metric="mae",
                 loss_weights: Union[List[float], None] = None,
                 mask_atom=False,
                 **__):
        """
        Loss function to deal with the loss calculation logic
        it will calculate transfer energies indirectly
        :param target_names: names of the target labels for loss calculation
        """
        self.loss_metric = loss_metric
        self.loss_weights = loss_weights if loss_weights is not None else [1.0] * len(target_names)
        self.mask_atom = mask_atom
        assert self.loss_metric in tags.loss_metrics

        self.target_names = deepcopy(target_names)
        self.num_targets = len(self.target_names)

    def __call__(self, model_output, data_batch, loss_detail=False, mol_lvl_detail=False):
        detail = {}

        prop_tgt, prop_pred = self.get_pred_target(model_output, data_batch)

        coe = 1.
        mae_loss = torch.mean(torch.abs(prop_pred - prop_tgt), dim=0, keepdim=True)
        mse_loss = torch.mean((prop_pred - prop_tgt) ** 2, dim=0, keepdim=True)
        rmse_loss = torch.sqrt(mse_loss)

        loss_weights = torch.as_tensor(self.loss_weights).to(prop_pred.device)  # e.g. tensor([1., 10.])
        if self.loss_metric == "mae" or self.loss_metric == "combined_mae":
            total_loss = (loss_weights * (coe * mae_loss)).sum()
        elif self.loss_metric == "mse":
            total_loss = (loss_weights * (coe * mse_loss)).sum()
        elif self.loss_metric == "rmse":
            total_loss = (loss_weights * (coe * rmse_loss)).sum()
        else:
            raise ValueError("Invalid total loss: " + self.loss_metric)

        if loss_detail:
            # record details including MAE, RMSE, Difference, etc..
            # It is required while valid and test step but not required in training
            for i, name in enumerate(self.target_names):
                detail["MAE_{}".format(name)] = mae_loss[:, i].item()
                detail["MSE_{}".format(name)] = mse_loss[:, i].item()
            if mol_lvl_detail:
                detail["PROP_PRED"] = prop_pred.detach().cpu()
                detail["PROP_TGT"] = prop_tgt.detach().cpu()
        else:
            detail = None

        if loss_detail:
            # n_units: number of molecules
            detail["n_units"] = data_batch.N.shape[0]
            if self.mask_atom:  # fixme: it's better to use the batch data shape in the calculation of loss
                detail['n_units'] = data_batch.mask.sum().item()
            else:
                detail["n_units"] = data_batch.N.sum().item()
            detail["ATOM_Z"] = data_batch.Z.detach().cpu()
            return total_loss, detail
        else:
            return total_loss

    def get_processed_pred(self, model_output: dict, data_batch):
        # multi-task prediction
        prop_pred = model_output["atom_prop"]
        if self.mask_atom:
            mask = data_batch.mask.bool()
            prop_pred = prop_pred[mask, :]
        return prop_pred

    def get_pred_target(self, model_output: dict, data_batch):
        """
        Get energy target from data_batch
        Solvation energy is in kcal/mol but gas/water/octanol energy is in eV
        """
        prop_pred = self.get_processed_pred(model_output, data_batch)
        prop_tgt = torch.cat([getattr(data_batch, name).view(-1, 1) for name in self.target_names], dim=-1)
        assert prop_pred.shape[-1] == self.num_targets
        if self.mask_atom:
            mask = data_batch.mask.bool()
            prop_tgt = prop_tgt[mask, :]
        return prop_tgt, prop_pred
