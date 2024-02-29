import logging
from math import ceil

import torch
from Networks.ResidualLayer import ResidualLayer
from Networks.ActivationFns import activation_getter
from Networks.Interaction_module import InteractionModule
from utils.utils_functions import floating_type


class OutputLayer(torch.nn.Module):
    """
    The output layer(red one in paper) of PhysNet
    """

    def __init__(self, F, n_output, n_res_output, activation, zero_last_linear=True, n_read_out=0, batch_norm=False,
                 dropout=False, bias=False):
        self.batch_norm = batch_norm
        super().__init__()
        self.n_res_output = n_res_output
        self.n_read_out = n_read_out
        for i in range(n_res_output):
            self.add_module('res_layer' + str(i), ResidualLayer(F, activation, batch_norm=batch_norm, dropout=dropout))

        last_dim = F
        for i in range(n_read_out):
            this_dim = ceil(last_dim/2)
            read_out_i = torch.nn.Linear(last_dim, this_dim)
            last_dim = this_dim
            self.add_module('read_out{}'.format(i), read_out_i)
            if self.batch_norm:
                self.add_module("bn_{}".format(i), torch.nn.BatchNorm1d(last_dim, momentum=1.))

        self.lin = torch.nn.Linear(last_dim, n_output, bias=bias)
        if zero_last_linear:
            self.lin.weight.data.zero_()
        else:
            logging.info("Output layer not zeroed, make sure you are doing classification.")
        if self.batch_norm:
            self.bn_last = torch.nn.BatchNorm1d(last_dim, momentum=1.)

        self.activation = activation_getter(activation)

    def forward(self, x):
        tmp_res = x

        for i in range(self.n_res_output):
            tmp_res = self._modules['res_layer' + str(i)](tmp_res)
        out = tmp_res

        for i in range(self.n_read_out):
            if self.batch_norm:
                out = self._modules["bn_{}".format(i)](out)
            a = self.activation(out)
            out = self._modules['read_out{}'.format(i)](a)

        if self.batch_norm:
            out = self.bn_last(out)
        out = self.activation(out)
        out = self.lin(out)
        return out


class PhysModule(torch.nn.Module):
    """
    Main module in PhysNet. Output layer is not included in this module.
    """

    def __init__(self, F, K, n_res_atomic, n_res_interaction, activation, batch_norm, dropout):
        super().__init__()
        self.interaction = InteractionModule(F=F, K=K, n_res_interaction=n_res_interaction, activation=activation,
                                             batch_norm=batch_norm, dropout=dropout).type(floating_type)
        self.n_res_atomic = n_res_atomic
        for i in range(n_res_atomic):
            self.add_module('res_layer' + str(i), ResidualLayer(F, activation, batch_norm=batch_norm, dropout=dropout))

    def forward(self, input_dict):
        x = input_dict["vi"]
        edge_index = input_dict["edge_index"]
        edge_attr = input_dict["edge_attr"]["rbf"]
        interacted_x, _ = self.interaction(x, edge_index, edge_attr)
        for i in range(self.n_res_atomic):
            interacted_x = self._modules['res_layer' + str(i)](interacted_x)

        return {"vi": interacted_x}