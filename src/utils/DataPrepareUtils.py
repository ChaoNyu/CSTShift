import torch
from utils.utils_functions import get_device


def _get_index_from_matrix(num, previous_num):
    """
    get the edge index compatible with torch_geometric message passing module
    eg: when num = 3, will return:
    [[0, 0, 0, 1, 1, 1, 2, 2, 2]
    [0, 1, 2, 0, 1, 2, 0, 1, 2]]
    :param num:
    :param previous_num: the result will be added previous_num to fit the batch
    :return:
    """
    matrix_to_index_map = {} 
    if num in matrix_to_index_map.keys():
        return matrix_to_index_map[num] + previous_num
    else:
        index = torch.LongTensor(2, num * num).to(get_device())
        index[0, :] = torch.cat([torch.zeros(num, device=get_device()).long().fill_(i) for i in range(num)], dim=0)
        index[1, :] = torch.cat([torch.arange(num, device=get_device()).long() for _ in range(num)], dim=0)
        mask = (index[0, :] != index[1, :])
        matrix_to_index_map[num] = index[:, mask]
        return matrix_to_index_map[num] + previous_num


def cal_edge(R, N, prev_N, edge_index, cal_coulomb=True):
    """
    calculate edge distance from edge_index;
    if cal_coulomb is True, additional edge will be calculated without any restriction
    :param cal_coulomb:
    :param prev_N:
    :param edge_index:
    :param R:
    :param N:
    :return:
    """
    if cal_coulomb:
        '''
        IMPORTANT: DO NOT use num(tensor) itself as input, which will be regarded as dictionary key in this function,
        use int value(num.item())
        Using tensor as dictionary key will cause unexpected problem, for example, memory leak
        '''
        coulomb_index = torch.cat(
            [_get_index_from_matrix(num.item(), previous_num) for num, previous_num in zip(N, prev_N)], dim=-1)
        points1 = R[coulomb_index[0, :], :]
        points2 = R[coulomb_index[1, :], :]
        coulomb_dist = torch.sum((points1 - points2) ** 2, keepdim=True, dim=-1)
        coulomb_dist = torch.sqrt(coulomb_dist)

    else:
        coulomb_dist = None
        coulomb_index = None

    short_range_index = edge_index
    points1 = R[edge_index[0, :], :]
    points2 = R[edge_index[1, :], :]
    short_range_dist = torch.sum((points1 - points2) ** 2, keepdim=True, dim=-1)
    short_range_dist = torch.sqrt(short_range_dist)
    return coulomb_dist, coulomb_index, short_range_dist, short_range_index


def sort_edge(edge_index):
    """
    sort the target of edge to be sequential, which may increase computational efficiency later on when training
    :param edge_index:
    :return:
    """
    arg_sort = torch.argsort(edge_index[1, :])
    return edge_index[:, arg_sort]


def remove_bonding_edge(all_edge_index, bond_edge_index):
    """
    Remove bonding idx_name from atom_edge_index to avoid double counting
    :param all_edge_index:
    :param bond_edge_index:
    :return:
    """
    mask = torch.zeros(all_edge_index.shape[-1]).bool().fill_(False).type(all_edge_index.type())
    len_bonding = bond_edge_index.shape[-1]
    for i in range(len_bonding):
        same_atom = (all_edge_index == bond_edge_index[:, i].view(-1, 1))
        mask += (same_atom[0] & same_atom[1])
    remain_mask = ~ mask
    return all_edge_index[:, remain_mask]


def calculate_edges(data, cutoff, record_long_range):
    """
    edge calculation
    """
    edge_index = torch.zeros(2, 0).long()
    dist, full_edge, _, _ = cal_edge(data.R, [data.N], [0], edge_index, cal_coulomb=True)
    dist = dist.cpu()
    full_edge = full_edge.cpu()

    data.BN_edge_index = full_edge[:, (dist < cutoff).view(-1)]

    if record_long_range:
        data.L_edge_index = remove_bonding_edge(full_edge, data.BN_edge_index)

    # sort edge idx_name
    data.BN_edge_index = sort_edge(data.BN_edge_index)
    _edge_index = data.BN_edge_index

    for bond_type in ['B', 'N', 'L', 'BN']:
        _edge_index = getattr(data, bond_type + '_edge_index', False)
        if _edge_index is not False:
            setattr(data, 'num_' + bond_type + '_edge', torch.zeros(1).long() + _edge_index.shape[-1])

    return data
