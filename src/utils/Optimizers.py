import torch
from torch.optim.swa_utils import AveragedModel
from utils.utils_functions import get_device


class EmaAmsGrad(torch.optim.Adam):
    def __init__(self, training_model: torch.nn.Module, lr=1e-3, betas=(0.9, 0.99),
                 eps=1e-8, weight_decay=0, ema=0.999, shadow_dict=None, params=None):
        if params is None:
            params = training_model.parameters()
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad=True)
        # for initialization of shadow model
        self.shadow_dict = shadow_dict
        self.ema = ema
        self.training_model = training_model

        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            return ema * averaged_model_parameter + (1 - ema) * model_parameter

        def avg_fn_deactivated(averaged_model_parameter, model_parameter, num_averaged):
            return model_parameter

        self.deactivated = (ema < 0)
        self.shadow_model = AveragedModel(training_model, device=get_device(),
                                          avg_fn=avg_fn_deactivated if self.deactivated else avg_fn)

    def step(self, closure=None):
        loss = super().step(closure)
        if self.shadow_model.n_averaged == 0 and self.shadow_dict is not None:
            self.shadow_model.module.load_state_dict(self.shadow_dict, strict=False)
            self.shadow_model.n_averaged += 1
        else:
            self.shadow_model.update_parameters(self.training_model)

        return loss