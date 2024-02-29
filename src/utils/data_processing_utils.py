import re
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from utils.DataPrepareUtils import calculate_edges
import torch
from tqdm import tqdm


def get_eigenvalue_from_line(line, decimal=4):
    """
    return 3 eigenvalues from log line. In log files eigenvalues have four decimal places.
    Normal line e.g. "Eigenvalues: -10040.7946  -673.7200   242.5350"
    More examples:
    Eigenvalues:-10040.7946  -673.7200   242.5350
    Eigenvalues:  -1700.9091   244.4306136356.7178
    """
    pattern = re.compile(r"Eigenvalues:\s*(-?\d+\.\d{%d})\s*(-?\d+\.\d{%d})\s*(-?\d+\.\d{%d})" % (decimal, decimal, decimal))  # (-?\d+\.\d+) is often used to catch float number
    match = pattern.search(line)
    if match:
        eigenvalues = [float(i) for i in match.groups()]
        return eigenvalues
    else:
        return None


def get_N_atom_from_log(log_file):
    """get the atom number from Gaussian log file with given path"""
    # if log_file is str
    if isinstance(log_file, str):
        with open(log_file, "r") as f:
            lines = f.readlines()
    else:  # log_file is a list of lines
        lines = log_file
    try:
        for line in lines:
            if line.strip(' ').startswith('NAtoms='):
                N_atom = [i for i in line.strip(' ').split(' ') if i][1]
                return int(N_atom)
    except:
        print("Can't get N_atom from log file: {}".format(log_file))
        return None


def get_charge_from_log(log_file):
    """get a list of atomic Mulliken charge from Gaussian log file with given path"""
    if isinstance(log_file, str):
        with open(log_file, "r") as f:
            lines = f.readlines()
    else:  # log_file is a list of lines
        lines = log_file
    N_atom = get_N_atom_from_log(log_file)
    charges = []
    try:
        for idx, line in enumerate(lines):
            if line.strip(' ').startswith('Mulliken charges:'):
                charge_start_line = idx + 2
                break
        for atom_idx in range(int(N_atom)):
            charges.append(float([i for i in lines[charge_start_line+atom_idx].strip(' ').split(' ') if i][2]))
    except:
        print("Can't get charge from log file: {}".format(log_file))
    return charges


def get_raw_qmd_feature_from_log(log_file):
    """get raw qmd feature from Gaussian log file with given path"""
    if isinstance(log_file, str):
        with open(log_file, "r") as f:
            lines = f.readlines()
    else:  # log_file is a list of lines
        lines = log_file
    N_atom = get_N_atom_from_log(log_file)
    raw_qmd_features = []
    try:
        for idx, line in enumerate(lines):
            if line.strip(' ').startswith('SCF GIAO Magnetic shielding tensor (ppm):'):
                nmr_start = idx + 5  # read last line of "Eigenvalues: ..."
                break
        for line_number in range(N_atom):
            line = lines[nmr_start + line_number * 5]
            raw_qmd_features.append(get_eigenvalue_from_line(line))
    except:
        print("Can't get raw qmd feature from log file: {}".format(log_file))
    return raw_qmd_features


def get_pos_z_from_log(log_file):
    """get a list of atomic positions and atom type from Gaussian log file with given path"""
    if isinstance(log_file, str):
        with open(log_file, "r") as f:
            lines = f.readlines()
    else:  # log_file is a list of lines
        lines = log_file
    N_atom = get_N_atom_from_log(log_file)
    positions = []
    atom_types = []
    try:
        for idx, line in enumerate(lines):
            if line.strip(' ').startswith('Standard orientation:'):
                pos_start_line = idx + 5
                break
        for atom_idx in range(int(N_atom)):
            line = lines[pos_start_line+atom_idx].split()
            # e.g. ['1', '6', '0'(atomic type), '0.000000', '0.000000', '0.000000']
            positions.append([float(i) for i in line[3:6]])
            atom_types.append(int(line[1]))
        assert lines[pos_start_line + atom_idx + 1].strip(' ').startswith('----')  # assert last line is -----------
    except:
        print("Can't get position and atom type from log file: {}".format(log_file))
    return positions, atom_types


def get_all_data_from_nmr_log(log_file, extracting_methods={'raw_qmd_features': get_raw_qmd_feature_from_log}):
    """extract info from nmr log files calculated by Gaussian"""
    N_atom = get_N_atom_from_log(log_file)
    positions, atom_types = get_pos_z_from_log(log_file)
    data = {}
    try:
        for method_name, method in extracting_methods.items():
            data[method_name] = method(log_file)
        data['R'] = positions
        data['Z'] = atom_types
        data['N'] = N_atom
    except:
        print("Can't get all data from log file: {}".format(log_file))
    return data


def get_atom_y_and_mask(shift_dict, atom_num):
    """get the atom_y (shift) and mask for molecules from Jonas data"""
    atom_y = np.zeros(atom_num, dtype=np.float32)
    mask = np.zeros(atom_num, dtype=np.float32)
    for atom_idx, shift in shift_dict.items():
        atom_y[atom_idx] = shift
        mask[atom_idx] = 1.0
    return atom_y, mask


def combine_mol_dict_to_dataset(dict_mols, save_path):
    """combine a list of mol dict to a dataset"""
    data_list = []
    for mol in tqdm(dict_mols):
        mol_info = Data(**mol)
        mol_info_w_edge = calculate_edges(mol_info, cutoff=10.0, record_long_range=True)
        data_list.append(mol_info_w_edge)
    # save the processed dataset
    processed_data, slices, _ = collate(data_list[0].__class__, data_list=data_list, increment=False, add_batch=False)
    torch.save((processed_data, slices), save_path)
