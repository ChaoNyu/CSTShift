from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
from rdkit import Chem
from collections import Counter
from rdkit.Chem import rdDetermineBonds
from utils.cheshire_dataset import read_combined_file
import re

JONAS_REPO = '/scratch/ch3859/nmr_related_repo/jonas_shift/'
JONAS_C_DATA = JONAS_REPO + 'graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle'
JONAS_H_DATA = JONAS_REPO + 'graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.1H.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.1.mol_dict.pickle'


def get_jonas_split(dataset, data_type='c', valid_ratio=0, seed=42):
    """
    get the training/test split according to jonas dataset splitting rules. 
    dataset: The input dataset must have the attributes of molecule_id from jonas dataset
    data_type: 'c' or 'h'
    valid_ratio: 0 means no validation set
    return: 
    """
    jonas_data = pd.read_pickle(JONAS_C_DATA if data_type == 'c' else JONAS_H_DATA)
    jonas_train = jonas_data['train_df']['molecule_id'].unique()
    jonas_test = jonas_data['test_df']['molecule_id'].unique()
    train_idx = []
    test_idx = []
    valid_idx = []
    for idx, mole_id in enumerate(dataset['molecule_id']):
        mole_id = mole_id.item()
        if mole_id in jonas_train:
            train_idx.append(idx)
        elif mole_id in jonas_test:
            test_idx.append(idx)
    if valid_ratio > 0:
        train_idx, valid_idx = train_test_split(train_idx, test_size=valid_ratio, random_state=seed)
    return train_idx, valid_idx, test_idx


def filter_splits(data, slices, splits, criteria, save_path=None, **kwargs):
    """
    filter splits according to criteria
    splits: {'train_index': [], 'val_index': [], 'test_index': []}, elements are tensor
    """
    new_splits = {}
    for split_name, split in splits.items():
        new_splits[split_name] = [idx for idx in split if criteria(data, slices, idx, **kwargs)]
    if save_path is not None:
        torch.save(new_splits, save_path)
    return new_splits


def get_atom_symbol_from_number(atom_number):
    """get the atomic symbol from atomic number"""
    return Chem.GetPeriodicTable().GetElementSymbol(atom_number)


def is_neutral_by_determine_bonds(data, slices, idx):
    """use rdDetermineBonds to check if the molecule is neutral"""
    atom_begin_idx = slices['Z'][idx]
    atom_end_idx = slices['Z'][idx+1]
    pos = data['R'][atom_begin_idx:atom_end_idx]
    atom_types = data['Z'][atom_begin_idx:atom_end_idx]
    # turn atom_types into atomic symbols
    atom_symbols = [get_atom_symbol_from_number(i.item()) for i in atom_types]
    try:
        mol = get_rdmol_from_pos(pos, atom_symbols, add_bonds=True)
        return True
    except:
        return False


def is_in_given_element_set(data, slices, idx, element_set=[1,6,7,8,9,15,16,17]):
    """"check if the molecule contains atom in a given element set"""
    atom_types = data['Z'][slices['Z'][idx]:slices['Z'][idx+1]]
    return all([i in element_set for i in atom_types])


def is_raw_qmd_feat_outlier(data, slices, idx, atomic_type=6, higher_bound=1000, lower_bound=-1000):
    """check if the molecule is an outlier due to too high or too low raw qmd features"""
    raw_qmd_features = data['raw_qmd_features'][slices['raw_qmd_features'][idx]:slices['raw_qmd_features'][idx+1]]
    atom_types = data['Z'][slices['Z'][idx]:slices['Z'][idx+1]]
    atom_idx = np.where(atom_types == atomic_type)[0]
    if len(atom_idx) == 0:
        return False
    has_higher_outlier = (raw_qmd_features[atom_idx] > higher_bound).any().item()
    has_lower_outlier = (raw_qmd_features[atom_idx] < lower_bound).any().item()
    return has_higher_outlier or has_lower_outlier


def check_mol_sanity(mol):
    """check if a mol instance is valid through the sanity check of rdkit"""
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False


def count_elements_from_rdmol(mol):
    """count the number of each elements in a rdkit mol instance"""
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Counter(elements)


def get_rdmol_from_pos(coords, atom_symbols, add_bonds=False):
    """get rdkit mol instance from 3D atomic positions and atom types"""
    xyz_str = str(len(coords)) + '\n\n'
    for atom_symbol, coord in zip(atom_symbols, coords):
        xyz_str += f"{atom_symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"
    # turn xyz_str into a rdkit mol object
    mol = Chem.MolFromXYZBlock(xyz_str)
    if add_bonds:
        # add bonds according to the positions
        rdDetermineBonds.DetermineBonds(mol,charge=0)
    return mol


def label_equal_H(mol):
    """
    label the H connection to heavy atoms. 2 Hs share the same label if connected to the same heavy atom
    input: mol: rdkit mol instance with H
    return: a list of labels for each atom
    """
    atom_labels = []
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            neighbors = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors() if neighbor.GetAtomicNum() != 1]
            assert len(neighbors) == 1
            atom_labels.append(neighbors[0])
        else:
            atom_labels.append(idx)
    return atom_labels


def add_H_neighbor_labels(dataset, slices, jonas_data):
    """add H_neighbor labels to dataset using label_equal_H"""
    if 'train_df' in jonas_data.keys():
        jonas_data = pd.concat([jonas_data['train_df'], jonas_data['test_df']])
    H_neighbor_labels = torch.zeros_like(dataset['Z'])
    for idx, mol_id in enumerate(dataset['molecule_id']):
        mol = jonas_data['rdmol'][jonas_data['molecule_id'] == mol_id.item()].values[0]
        H_neighbor_labels[slices['Z'][idx]:slices['Z'][idx+1]] = torch.tensor(label_equal_H(mol))
    dataset['H_neighbor_labels'] = H_neighbor_labels
    slices['H_neighbor_labels'] = slices['Z']
    return dataset, slices


def add_H_neighbor_labels_from_rdmols(dataset, slices, rdmols, file_names):
    """add H_neighbor labels to dataset using label_equal_H"""
    H_neighbor_labels = torch.zeros_like(dataset['Z'])
    assert len(rdmols) == dataset['N'].shape[0]
    for mol_idx in range(len(dataset['N'])):
        atom_begin_idx = slices['Z'][mol_idx]
        atom_end_idx = slices['Z'][mol_idx+1]
        mol_file_name = dataset['mol_file_name'][mol_idx]
        # find the corresponding rdmol
        rdmol_idx = file_names.index(mol_file_name)
        mol = rdmols[rdmol_idx]
        if mol is None: continue  # skip if the mol is None
        H_neighbor_labels[atom_begin_idx:atom_end_idx] = torch.tensor(label_equal_H(mol))
    dataset['H_neighbor_labels'] = H_neighbor_labels
    slices['H_neighbor_labels'] = slices['Z']
    return dataset, slices


def add_equal_H_qmd_features(dataset, slices):
    """add equal_H_qmd_features to the Hydrogen dataset, where the H atoms with the same label are assigned the average value of raw_qmd_features"""
    equal_H_qmd_features = dataset['raw_qmd_features'].clone().detach()
    for mol_idx in range(len(dataset['N'])):
        atom_begin_idx = slices['Z'][mol_idx]
        atom_end_idx = slices['Z'][mol_idx+1]
        H_neighbor_labels = dataset['H_neighbor_labels'][atom_begin_idx:atom_end_idx]
        raw_qmd_features = dataset['raw_qmd_features'][atom_begin_idx:atom_end_idx]
        Z = dataset['Z'][atom_begin_idx:atom_end_idx]
        for atom_idx in range(atom_begin_idx, atom_end_idx):
            if dataset['Z'][atom_idx] != 1: continue
            label = dataset['H_neighbor_labels'][atom_idx].item()
            equal_H = torch.logical_and(H_neighbor_labels == label, Z == 1)
            equal_H_qmd_features[atom_idx] = raw_qmd_features[equal_H].mean(dim=0)
    dataset['equal_H_qmd_features'] = equal_H_qmd_features
    slices['equal_H_qmd_features'] = slices['raw_qmd_features']
    return dataset, slices


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


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def merge_duplicated_moles_by_linear_model(dup_moles, model, avg_raw_qmd_features):
    """
    merge duplicated molecules using a linear model fit by the non-duplicated moles
    this method might avoid the duplication caused by wrong atomic assignments or different solvents
    The best mole is selected by the lowest error MAE between the predicted and true chemical shifts on all atoms
    return: the merged molecule or None if it's not mergable
    """
    assert len(dup_moles) > 1
    lin_pred = model.predict(avg_raw_qmd_features)
    mae = [np.mean(np.abs((lin_pred - mol['atom_y'].numpy())[mol['mask'] == 1])) for mol in dup_moles]
    best_mol_idx = np.argmin(mae)
    return dup_moles[best_mol_idx]


def get_raw_qmd_features_by_id(dataset, slices, mol_id):
    """
    get the atomic features and mask of a molecule by molecule_id defined by jonas dataset
    Note that raw_qmd_features is the same for different instances with the same molecule_id
    """
    mol_idx = torch.where(dataset['molecule_id'] == mol_id)[0][0]
    atom_begin_idx = slices['mask'][mol_idx]
    atom_end_idx = slices['mask'][mol_idx+1]
    raw_qmd_features = dataset['raw_qmd_features'][atom_begin_idx:atom_end_idx]
    return raw_qmd_features


def count_atom_number(dataset, slices):
    """count atom number and heavy atom number in saved dataset"""
    atom_num = []
    heavy_atom_num = []
    for mol_idx, n in enumerate(dataset['N']):
        atom_num.append(n.item())
        heavy_atom_num.append(np.sum(dataset['Z'][slices['Z'][mol_idx]:slices['Z'][mol_idx+1]].numpy() > 1))
    return atom_num, heavy_atom_num


def get_atom_mol_batch(dataset, slices):
    """get the atom_mol_batch"""
    atom_mol_batch = torch.zeros_like(dataset['mask'])
    for i in range(len(slices['mask']) - 1):
        atom_mol_batch[slices['mask'][i]:slices['mask'][i+1]] = i
    return atom_mol_batch


def read_combined_Gaussian_log(file_path, divider='Initial command'):
    """
    read the combined Gaussian log file
    return a list of blocks, each block is a list of lines
    """
    com_blocks = read_combined_file(file_path, divider)
    # remove the first block if 'normal termination' is not found
    if not check_normal_termination(com_blocks[0]):
        com_blocks.pop(0)
    return com_blocks


def save_linked_Gaussian_log_separately(file_path, save_dir, divider='Initial command'):
    """
    save the linked Gaussian log file separately
    """
    com_blocks = read_combined_file(file_path, divider)
    if not check_normal_termination(com_blocks[0]):
        # combine the first two blocks
        com_blocks[1] = com_blocks[0] + com_blocks[1]
        com_blocks.pop(0)

    for idx, block in enumerate(com_blocks):
        with open(save_dir + str(idx) + '.log', 'w') as f:
            f.writelines(block)


def read_atom_info_from_com_file(com_lines):
    """
    read atomic type and position from Gaussian com file
    return: list of atomic type and position
    """
    atom_info = []
    pattern = r'[-+]?\d*\.\d+|\d+'  # This pattern matches floating-point and integer numbers
    for line in com_lines:
        numbers = re.findall(pattern, line)
        if len(numbers) == 4:
            atom_info.append([int(numbers[0]), float(numbers[1]), float(numbers[2]), float(numbers[3])])
    return atom_info


def check_normal_termination(log_file):
    """check if the Gaussian log file is terminated normally"""
    # if log_file is str
    if isinstance(log_file, str):
        with open(log_file, "r") as f:
            lines = f.readlines()
    else:  # log_file is a list of lines
        lines = log_file
    for line in lines:
        if line.startswith(' Normal termination of Gaussian'):
            return True
    return False


def check_optimization_completed(log_file):
    """
    check if the optimization is completed with given path or block of lines"""
    # if log_file is str
    if isinstance(log_file, str):
        with open(log_file, "r") as f:
            lines = f.readlines()
    else:  # log_file is a list of lines
        lines = log_file
    for line in lines:
        if 'Optimization completed' in line:
            return True
    return False


def get_final_energy_from_opt_log(log_file):
    """
    get the final energy from Gaussian log file with given path or block of lines"""
    if not check_optimization_completed(log_file):
        return None
    
    # if log_file is str
    if isinstance(log_file, str):
        with open(log_file, "r") as f:
            lines = f.readlines()
    else:  # log_file is a list of lines
        lines = log_file
    try:
        scf_energy = []
        for line in lines:
            if line.strip(' ').startswith('SCF Done:'):
                energy = line.strip(' ').split(' ')[6]  # e.g. 'SCF Done:  E(RB3LYP) =  -10040.7946     A.U. after   10 cycles'
                scf_energy.append(float(energy))
        return scf_energy[-1]
    except:
        print("Can't get energy from log file: {}".format(log_file))
        return None
