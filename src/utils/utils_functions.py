import copy
import logging
import os
from datetime import datetime
import numpy as np
import torch
import subprocess

# floating type
floating_type = torch.double
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_device():
    # we use a function to get device for proper distributed training behaviour
    # return torch.device("cpu")
    return _device


def solv_num_workers():
    try:
        n_cpu_avail = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu_avail = None
    n_cpu = os.cpu_count()
    num_workers = n_cpu_avail if n_cpu_avail is not None else n_cpu
    return n_cpu_avail, n_cpu, num_workers


def _cutoff_fn(D, cutoff):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    x = D / cutoff
    x3 = x ** 3
    x4 = x3 * x
    x5 = x4 * x

    result = 1 - 6 * x5 + 15 * x4 - 10 * x3
    return result


def gaussian_rbf(D, centers, widths, cutoff, return_dict=False):
    """
    The rbf expansion of a distance
    Input D: matrix that contains the distance between to atoms
          K: Number of generated distance features
    Output: A matrix containing rbf expanded distances
    """

    rbf = _cutoff_fn(D, cutoff) * torch.exp(-widths * (torch.exp(-D) - centers) ** 2)
    if return_dict:
        return {"rbf": rbf}
    else:
        return rbf


def softplus_inverse(x):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    return torch.log(-torch.expm1(-x)) + x


# generates a random square orthogonal matrix of dimension dim
def square_orthogonal_matrix(dim=3, seed=None):
    random_state = np.random
    if seed is not None:  # allows to get the same matrix every time
        random_state.seed(seed)
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


# generates a random (semi-)orthogonal matrix of size NxM
def semi_orthogonal_matrix(N, M, seed=None):
    if N > M:  # number of rows is larger than number of columns
        square_matrix = square_orthogonal_matrix(dim=N, seed=seed)
    else:  # number of columns is larger than number of rows
        square_matrix = square_orthogonal_matrix(dim=M, seed=seed)
    return square_matrix[:N, :M]


# generates a weight matrix with variance according to Glorot initialization
# based on a random (semi-)orthogonal matrix
# neural networks are expected to learn better when features are decorrelated
# (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
# "Dropout: a simple way to prevent neural networks from overfitting",
# "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
def semi_orthogonal_glorot_weights(n_in, n_out, scale=2.0, seed=None):
    W = semi_orthogonal_matrix(n_in, n_out, seed=seed)
    W *= np.sqrt(scale / ((n_in + n_out) * W.var()))
    return torch.Tensor(W).type(floating_type).t()


def get_n_params(model, logger=None, only_trainable=False):
    """
    Calculate num of parameters in the model
    :param only_trainable: Only count trainable
    :param logger:
    :param model:
    :return:
    """
    result = ''
    counted_params = []
    for name, param in model.named_parameters():
        if not (only_trainable and not param.requires_grad):
            if logger is not None:
                logger.info('{}: {}'.format(name, param.data.shape))
            result = result + '{}: {}\n'.format(name, param.data.shape)
            counted_params.append(param)
    return sum([x.nelement() for x in counted_params]), result


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def option_solver(option_txt, type_conversion=False, return_base=False):
    option_base = option_txt.split('[')[0]
    if len(option_txt.split('[')) == 1:
        result = {}
    else:
        # option_txt should be like :    '[n_read_out=2,other_option=value]'
        # which will be converted into a dictionary: {n_read_out: 2, other_option: value}
        option_txt = option_txt.split('[')[1]
        option_txt = option_txt[:-1]
        result = {argument.split('=')[0].strip(): argument.split('=')[1].strip()
                  for argument in option_txt.split(',')}
        if type_conversion:
            for key in result.keys():
                value_final = copy.copy(result[key])
                try:
                    tmp = float(value_final)
                    result[key] = tmp
                except ValueError:
                    pass

                try:
                    tmp = int(value_final)
                    result[key] = tmp
                except ValueError:
                    pass

                if result[key] in ["True", "False"]:
                    result[key] = (result[key] == "True")
    if return_base:
        return option_base, result
    else:
        return result


def get_folder_name(prefix="", suffix=""):
    """
    get the folder name as prefix + current time + suffix
    current time format: %y%m%d-%H%M%S
    If folder exists, add an additional number to the end, e.g. prefix + current time + suffix + _1, _2, ...
    """
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')[2:]
    folder_name = prefix + current_time + suffix
    if os.path.exists(folder_name):
        for i in range(1, 10):
            folder_name = prefix + current_time + suffix + "_" + str(i)
            if not os.path.exists(folder_name):
                break
    if os.path.exists(folder_name):
        raise ValueError("Folder already exists! Too many folders with the same name!")
    os.makedirs(folder_name)
    return folder_name


def remove_handler(log=None):
    if log is None:
        log = logging.getLogger()
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    return


def validate_index(train_index, val_index, test_index):
    # make sure the indexes are legit without overlapping, etc...
    train_size = train_index.shape[0]
    train_index_set = set(train_index.numpy().tolist())
    assert train_size == len(train_index_set), f"{train_size}, {len(train_index_set)}"

    val_size = val_index.shape[0]
    val_index_set = set(val_index.numpy().tolist())
    assert val_size == len(val_index_set), f"{val_size}, {len(val_index_set)}"
    assert len(train_index_set.intersection(val_index_set)) == 0, "You have a problem :)"

    if test_index is not None:
        test_size = test_index.shape[0]
        test_index_set = set(test_index.numpy().tolist())
        assert test_size == len(test_index_set), f"{test_size}, {len(test_index_set)}"
        assert len(train_index_set.intersection(test_index_set)) == 0, "You have a problem :)"
    else:
        test_size = None

    return train_size, val_size, test_size


def get_train_atomic_idx(data_provider, train_mol_idx, mask_atom=False):
    """get train_atomic_idx from mole idx."""
    train_atomic_idx = [torch.arange(data_provider.slices['Z'][start_idx], data_provider.slices['Z'][start_idx + 1]) for start_idx in train_mol_idx]
    train_atomic_idx = torch.cat(train_atomic_idx, dim=0)
    if mask_atom:
        train_atomic_idx = [i for i in train_atomic_idx if data_provider.data['mask'][i] == 1.0]
        train_atomic_idx = torch.tensor(train_atomic_idx)
    return train_atomic_idx


def get_ext_atom_feat_mean_std(data_provider, train_mol_idx, feat, ext_feat_dim, norm_method):
    """
    get the mean and std of external atom features. 
    These mean and std will be stored in the model so it should not be written as the transform for the dataset.
    """
    ext_atom_feature = data_provider.data.get(feat)
    z = data_provider.data.get('Z')
    assert ext_feat_dim == ext_atom_feature.shape[-1]
    train_atomic_idx = get_train_atomic_idx(data_provider, train_mol_idx, mask_atom=False)

    ext_atom_feat_mean_std = {'elements': z.unique()}
    assert ext_atom_feat_mean_std['elements'].tolist() == z[train_atomic_idx].unique().tolist()  # training set needs to contain all elements to make a meaningful embedding
    ext_atom_feat_mean_std['mean'] = torch.zeros(len(ext_atom_feat_mean_std['elements']), ext_feat_dim)
    ext_atom_feat_mean_std['std'] = torch.ones(len(ext_atom_feat_mean_std['elements']), ext_feat_dim)
    if norm_method == 'all':
        ext_atom_feat_mean_std['mean'][:] = torch.mean(ext_atom_feature[train_atomic_idx], dim=0) 
        ext_atom_feat_mean_std['std'][:] = torch.std(ext_atom_feature[train_atomic_idx], dim=0)
    else:  # norm by elements
        for idx, element in enumerate(ext_atom_feat_mean_std['elements']):
            element_idx = z[train_atomic_idx] == element
            ext_atom_feat_mean_std['mean'][idx] = torch.mean(ext_atom_feature[train_atomic_idx][element_idx], dim=0)
            ext_atom_feat_mean_std['std'][idx] = torch.std(ext_atom_feature[train_atomic_idx][element_idx], dim=0)
    return ext_atom_feat_mean_std


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def get_split(file_path=None, train_index=None, valid_index=None, test_index=None, all_test=0, toy_split=0, save=False):
    """get the split of dataset"""
    # read from file_path if file exists
    if file_path and os.path.exists(file_path):
        splits = torch.load(file_path)
    elif train_index is not None and valid_index is not None and test_index is not None:
        splits = {
            "train_index": torch.as_tensor(train_index),
            "valid_index": torch.as_tensor(valid_index),
            "test_index": torch.as_tensor(test_index)
        }
    elif all_test > 0:
        splits = {
            "train_index": torch.as_tensor([]),
            "valid_index": torch.as_tensor([]),
            "test_index": torch.as_tensor(list(range(all_test)))
        }
    else:
        raise ValueError("Invalid split setting.")
    if save:
        torch.save(splits, file_path)
    if toy_split > 0:
        # use the amount of toy_split as the size of train_index, remain the original valid_index and test_index
        splits["train_index"] = splits["train_index"][:toy_split]
    return splits


def avg_iso_atoms(pred, tgt, atom_mol_batch):
    """
    calculated averaged prediction for atoms with the same tgt in the same molecule
    atom_mol_batch: the index of molecule, same shape as pred and tgt. e.g. [0,0,0,1,1,1,1,2,2,2,2,2]
    """
    new_pred = torch.zeros_like(pred)
    all_mol_idx = torch.unique(atom_mol_batch)
    atom_idx = 0
    for m_idx in all_mol_idx:
        m_pred = pred[atom_mol_batch == m_idx]
        m_tgt = tgt[atom_mol_batch == m_idx]
        for t in m_tgt:
            new_pred[atom_idx] = m_pred[m_tgt == t].mean()
            atom_idx += 1
    return new_pred


def get_atom_mol_batch(data_N):
    """
    get atom_mol_batch used for avg_iso_atoms.
    """
    atom_mol_batch = []
    for i, n in enumerate(data_N):
        atom_mol_batch.extend([i] * n)
    return torch.tensor(atom_mol_batch)