import logging
import math

import torch
import torch.nn as nn

from Networks.PhysModule import PhysModule, OutputLayer
from Networks.EmbeddingLayer import EmbeddingLayer
from utils.utils_functions import floating_type, softplus_inverse, gaussian_rbf


class sPhysNet(nn.Module):
    """
    The simplified version of PhysNet
    """
    def __init__(self,
                 n_module=2,
                 n_atom_embedding=95,
                 n_feature=512,
                 n_output=1,
                 n_phys_atomic_res=1,
                 n_phys_interaction_res=1,
                 n_phys_output_res=1,
                 normalize=True,
                 expansion_n=64,
                 expansion_dist=10.0,
                 last_lin_bias=False,
                 mean_atom=None,
                 std_atom=None,
                 target_names=None,
                 batch_norm=False,
                 dropout=False,
                 **kwargs):
        """
        :param n_module: number of PhysModule
        :param n_atom_embedding: number of atoms to embed, usually set to 95
        :param n_feature: number of features for each atom
        :param n_output: number of output
        :param n_phys_atomic_res: number of atomic residual blocks in PhysModule
        :param n_phys_interaction_res: number of interaction residual blocks in PhysModule
        :param n_phys_output_res: number of output residual blocks in PhysModule
        :param normalize: whether to normalize the output
        :param expansion_n: number of Gaussian RBF expansion
        :param expansion_dist: distance for Gaussian RBF expansion
        :param last_lin_bias: whether to use bias in the last linear layer
        :param mean_atom: mean of atom features
        :param std_atom: std of atom features
        :param target_names: names of targets
        :param batch_norm: whether to use batch normalization
        :param dropout: whether to use dropout
        """
        super().__init__()
        self.logger = logging.getLogger()
        self.target_names = target_names
        self.num_targets = len(target_names)
        self.expansion_info = {'n': expansion_n, 'dist': expansion_dist}
        self.last_lin_bias = last_lin_bias
        self.n_atom_embedding = n_atom_embedding
        self.n_feature = n_feature
        self.n_module = n_module
        self.normalize = normalize
        self.n_phys_output_res = n_phys_output_res
        self.n_phys_interaction_res = n_phys_interaction_res
        self.n_phys_atomic_res = n_phys_atomic_res
        self.n_output = n_output

        # registering necessary parameters for expansions
        feature_dist = torch.as_tensor(self.expansion_info['dist']).type(floating_type)
        self.register_parameter('cutoff', torch.nn.Parameter(feature_dist, False))
        centers = softplus_inverse(torch.linspace(1.0, math.exp(-feature_dist), self.expansion_info['n']))
        centers = torch.nn.functional.softplus(centers)
        self.register_parameter('centers', torch.nn.Parameter(centers, True))
        widths = [softplus_inverse((0.5 / ((1.0 - torch.exp(-feature_dist)) / self.expansion_info['n'])) ** 2)] * self.expansion_info['n']
        widths = torch.as_tensor(widths).type(floating_type)
        widths = torch.nn.functional.softplus(widths)
        self.register_parameter('widths', torch.nn.Parameter(widths, True))

        self.dist_calculator = nn.PairwiseDistance(keepdim=True)
        self.embedding_layer = EmbeddingLayer(n_atom_embedding, n_feature)

        # registering modules
        self.module_list = nn.ModuleList()
        for _ in range(self.n_module):
            this_module = PhysModule(F=n_feature,
                                     K=self.expansion_info['n'],
                                     n_res_atomic=n_phys_atomic_res,
                                     n_res_interaction=n_phys_interaction_res,
                                     activation='ssp',
                                     batch_norm=batch_norm,
                                     dropout=dropout)
            self.module_list.append(this_module)
        
        # register output layers
        self.output = OutputLayer(F=n_feature, n_output=n_output, n_res_output=n_phys_output_res, activation='ssp', 
                                  batch_norm=batch_norm, dropout=dropout, bias=last_lin_bias)

        if self.normalize:
            # Atom-wise shift and scale, used in PhysNet
            shift_matrix = torch.zeros(95, n_output).type(floating_type)
            scale_matrix = torch.zeros(95, n_output).type(floating_type).fill_(1.0)
            if mean_atom is not None:
                if isinstance(mean_atom, torch.Tensor):
                    shift_matrix[:, :] = mean_atom.view(1, -1)[:, :n_output]
                else:
                    shift_matrix[:, 0] = mean_atom
            if std_atom is not None:
                if isinstance(std_atom, torch.Tensor):
                    scale_matrix[:, :] = std_atom.view(1, -1)[:, :n_output]
                else:
                    scale_matrix[:, 0] = std_atom
            self.register_parameter('scale', torch.nn.Parameter(scale_matrix, requires_grad=True))
            self.register_parameter('shift', torch.nn.Parameter(shift_matrix, requires_grad=True))
 
    def normalize_output(self, out_sum, Z):
        # Atom-wise shifting and scale
        out_sum = self.scale[Z, :] * out_sum + self.shift[Z, :]
        return out_sum

    def calculate_expansion(self, R, edge_index):
        pair_dist = self.dist_calculator(R[edge_index[0, :], :], R[edge_index[1, :], :])
        expansions = gaussian_rbf(pair_dist, getattr(self, 'centers'),
                                             getattr(self, 'widths'),
                                             getattr(self, 'cutoff'),
                                             return_dict=True)
        return expansions

    def forward(self, data):
        R = data.R.type(floating_type)
        Z = data.Z
        edge_index = getattr(data, 'BN_edge_index', False)
        expansions = self.calculate_expansion(R, edge_index)
        vi_init = self.embedding_layer(Z)
        out_dict = {"vi": vi_init, "mji": None}
        
        for _module in self.module_list:
            out_dict["edge_index"] = edge_index
            out_dict["edge_attr"] = expansions
            out_dict = _module(out_dict)

        out_res = self.output(out_dict["vi"])
        if self.normalize:
            out_res = self.normalize_output(out_res, Z)

        return {"atom_prop": out_res}


class CSTShiftConcat(sPhysNet):
    """
    CSTShift model with concatenation between CST descriptors and atom embeddings. 
    Not directly used, only as the parent class
    """

    def __init__(self, 
                 ext_atom_feat_mean_std={},
                 ext_atom_features=None,
                 ext_atom_dim=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.ext_atom_dim = ext_atom_dim
        self.ext_atom_features = ext_atom_features
        
        if not ext_atom_feat_mean_std:
            assert ext_atom_dim > 0
            ext_atom_feat_mean_std['elements'] = torch.FloatTensor([1, 6, 7, 8, 9, 15, 16, 17])
            ext_atom_feat_mean_std['mean'] = torch.zeros(len(ext_atom_feat_mean_std['elements']), ext_atom_dim)
            ext_atom_feat_mean_std['std'] = torch.ones(len(ext_atom_feat_mean_std['elements']), ext_atom_dim)

        self.register_parameter('ext_atom_features_elements', 
                                nn.Parameter(ext_atom_feat_mean_std['elements'], requires_grad=False))
        self.register_parameter('ext_atom_features_mean', 
                                nn.Parameter(ext_atom_feat_mean_std['mean'], requires_grad=False))
        self.register_parameter('ext_atom_features_std', 
                                nn.Parameter(ext_atom_feat_mean_std['std'], requires_grad=False))


class CSTShiftEmb(CSTShiftConcat):
    """
    CSTShift with concatenation between CST descriptors and initial atom embeddings
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # overwrite the embedding layer with the new dimension setting
        self.embedding_layer = EmbeddingLayer(self.n_atom_embedding, self.n_feature - self.ext_atom_dim)
    
    def forward(self, data):
        R = data.R.type(floating_type)
        Z = data.Z
        edge_index = getattr(data, 'BN_edge_index', False)
        expansions = self.calculate_expansion(R, edge_index)
        vi_init = self.embedding_layer(Z)
        ext_atom_features = getattr(data, self.ext_atom_features).type(floating_type)
        for idx, element in enumerate(self.ext_atom_features_elements):
            # select the entries in ext_atom_features where the corresponding Z is equal to element, then use ext_atom_features_mean and ext_atom_features_std to normalize
            ext_atom_features[Z == element] = (ext_atom_features[Z == element] - self.ext_atom_features_mean[idx]) / self.ext_atom_features_std[idx]
        vi_init = torch.cat([vi_init, ext_atom_features], dim=-1)
        out_dict = {"vi": vi_init, "mji": None}
        for _module in self.module_list:
            out_dict["edge_index"] = edge_index
            out_dict["edge_attr"] = expansions
            out_dict = _module(out_dict)

        out_res = self.output(out_dict["vi"])
        if self.normalize:
            out_res = self.normalize_output(out_res, Z)

        return {"atom_prop": out_res}


class CSTShiftOut(CSTShiftConcat):
    """
    CSTShift model with concatenation between CST descriptors and output atom features
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # register the linear layer for the external atom features
        self.lin_ext_feat_in = torch.nn.Linear(self.n_feature + self.ext_atom_dim, self.n_feature)


    def forward(self, data):
        R = data.R.type(floating_type)
        Z = data.Z
        edge_index = getattr(data, 'BN_edge_index', False)
        expansions = self.calculate_expansion(R, edge_index)
        vi_init = self.embedding_layer(Z)
        out_dict = {"vi": vi_init, "mji": None}

        for _module in self.module_list:
            out_dict["edge_index"] = edge_index
            out_dict["edge_attr"] = expansions
            out_dict = _module(out_dict)

        output_ext_atom_features = getattr(data, self.ext_atom_features).type(floating_type)
        for idx, element in enumerate(self.ext_atom_features_elements):
            # select the entries in ext_atom_features where the corresponding Z is equal to element, then use ext_atom_features_mean and ext_atom_features_std to normalize
            output_ext_atom_features[Z == element] = (output_ext_atom_features[Z == element] - self.ext_atom_features_mean[idx]) / self.ext_atom_features_std[idx]
        vi = torch.cat((out_dict["vi"], output_ext_atom_features), dim=-1)
        vi = self.lin_ext_feat_in(vi)
        out_res = self.output(vi)
        if self.normalize:
            out_res = self.normalize_output(out_res, Z)

        return {"atom_prop": out_res}
