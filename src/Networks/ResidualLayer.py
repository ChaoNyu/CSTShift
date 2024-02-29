import torch.nn as nn

from Networks.ActivationFns import activation_getter
from utils.utils_functions import semi_orthogonal_glorot_weights, get_n_params


class ResidualLayer(nn.Module):
    """
    The residual layer defined in PhysNet
    """
    def __init__(self, F, activation, batch_norm=False, dropout=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.activation = activation_getter(activation)

        self.lin1 = nn.Linear(F, F)
        self.lin1.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin1.bias.data.zero_()
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(F, momentum=1.)

        self.lin2 = nn.Linear(F, F)
        self.lin2.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin2.bias.data.zero_()
        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(F, momentum=1.)

    def forward(self, x):
        x_res = x

        if self.batch_norm:
            x = self.bn1(x)

        x = self.activation(x)
        x = self.lin1(x)

        if self.batch_norm:
            x = self.bn2(x)

        x = self.activation(x)

        x = self.lin2(x)
        return x + x_res


if __name__ == '__main__':
    model = ResidualLayer(128)
    print(get_n_params(model))
