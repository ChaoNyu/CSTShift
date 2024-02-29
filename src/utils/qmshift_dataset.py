
from collections import defaultdict, Counter
from torch_geometric.data import InMemoryDataset
from rdkit.Chem.rdmolops import GetFormalCharge
import os
import tqdm
from typing import Callable, Optional
import torch
from nmr_dataset import MolData
import copy
import pandas as pd
import os.path as osp
from utils.data_processing_utils import get_atom_y_and_mask, get_all_data_from_nmr_log, merge_duplicated_moles, copy_mol_from_dataset, combine_mol_dict_to_dataset
import numpy as np


CHESHIRE_ID = [8453, 4742, 8967, 8966, 9614, 8464, 8977, 15897, 8731, 8631, 18238, 16069, 
               17996, 20813, 9173, 9558, 9559, 8791, 9564, 9567, 9824, 14945, 18273, 8050]
# CHESHIRE_ID is copied from Dongdong's notebooks. The method to get this list is not given.


def merge_duplicated_moles(dup_moles):
    """
    merge duplicated molecules if they share the same maximum set of nmr chemcial shift values
    dup_moles: a list of dict: {atom_y: [], mask: []}
    return: the merged molecule or None if it's not mergable

    e.g.
    mask1 = [0,1,1,0,0,0]
    mask2 = [0,1,1,1,1,0]
    mask3 = [0,0,0,1,1,1]
    """
    assert len(dup_moles) > 1
    # combine atom_y and mask into two np.array
    atom_y = np.stack([mol['atom_y'] for mol in dup_moles])
    mask = np.stack([mol['mask'] for mol in dup_moles])
    new_mask = np.zeros_like(mask[0])
    new_atom_y = np.zeros_like(atom_y[0])
    for atom_idx in range(atom_y.shape[1]):
        if np.all(mask[:, atom_idx] == 0): continue  # skip if all masks are 0
        # find if the atom_y value are the same for atoms with the same mask
        atom_y_same = np.unique(atom_y[:, atom_idx][mask[:, atom_idx] == 1]).shape[0] == 1
        if not atom_y_same: 
            return None
        else:
            new_mask[atom_idx] = 1
            new_atom_y[atom_idx] = atom_y[:, atom_idx][mask[:, atom_idx] == 1][0]
    # check if the moles have the same mask sum. If so, they are exactly the same molecules.
    if all(np.sum(mask, axis=1) == np.sum(new_mask)):
        print("Warning: the merged molecules are exactly the same.")
    return {'atom_y': new_atom_y, 'mask': new_mask}


def copy_mol_from_dataset(idx, dataset, slices, 
                          keys={'R', 'N', 'Z', 'atom_y', 'mask', 'molecule_id', 'raw_qmd_features'}):
    """copy mol instance from dataset with given index. Typically used for creating a new dataset from existing dataset
    the shape of tensor in dataset is (N) or (N, fix_number) (e.g. raw_qmd_features is (N, 3)) 
    """
    mol = {}
    for key in keys:
        mol[key] = dataset[key][slices[key][idx]:slices[key][idx+1]]
    return mol


class QMShiftDataset(InMemoryDataset):
    """
    Dataset combining the processed chemical shift data from Jonas 2019 and DFT calculated data from Gaussian log files.
    """
    def __init__(
        self,
        root: Optional[str] = None,
        nmr_log_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        self.nmr_log_path = nmr_log_path
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if not isinstance(self.data, MolData):
            self.data = MolData.from_data(self.data)
            self.slices["atom_mol_batch"] = copy.copy(self.slices["Z"])
            self.slices["sample_id"] = copy.copy(self.slices["N"])
        
    @property
    def raw_file_names(self):
        # download is disabled in this dataset because there's a huge number of log files
        return ['jonas_dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ['processed.pt']

    def download(self):
        # As the number of log files is huge, only the processed data is provided. 
        pass

    def process(self):
        jonas_dataset = pd.read_pickle(self.raw_file_names[0])
        jonas_dataset = pd.concat([jonas_dataset['train_df'], jonas_dataset['test_df']])

        data_list = []
        for i in tqdm(range(len(jonas_dataset))):
            # extract info from exp dataset
            molecule_id = jonas_dataset['molecule_id'][i]
            exp_shift = jonas_dataset['value'][i][0]
            charge = GetFormalCharge(jonas_dataset['rdmol'][i])
            nmr_log_file = self.nmr_log_path + '{}.log'.format(molecule_id)
            if not os.path.exists(nmr_log_file):
                continue

            # extract info from nmr_log
            mol_info = get_all_data_from_nmr_log(nmr_log_file)

            # conditions: no CHESHIRE_ID, charge 0, info in nmr_log
            if mol_info is None: continue
            if molecule_id in CHESHIRE_ID or charge != 0: continue

            # get atom_y and mask
            atom_y, mask = get_atom_y_and_mask(exp_shift, mol_info['N'])

            # pack the info together in torch format into a dict
            mol_info.update({'atom_y': atom_y, 'mask': mask, 'molecule_id': molecule_id})
            for attr in ['atom_y', 'R', 'raw_qmd_features', 'mask']:
                mol_info[attr] = torch.tensor(mol_info[attr]).to(torch.float32)  # todo: dtype
            for attr in ['Z', 'N', 'molecule_id']:
                mol_info[attr] = torch.tensor(mol_info[attr]).to(torch.long)
            data_list.append(mol_info)
        combine_mol_dict_to_dataset(data_list, self.processed_paths[0])


class MergedQMShiftDataset(InMemoryDataset):
    """
    There are duplications in the dataset where the same molecule has several nmr shift values. As not all the carbons/hydrogens are measured in each experiment, part of the duplication is caused by the lack of some exp values in certain experiments. Other duplication may be caused by experiments with different settings. 
    In this dataset, duplications are merged by keeping the maximum set of exp values for each molecule. 
    There are 95 duplicated sets of molecules in carbon dataset which have exactly the same nmr shift values. 9 in hydrogen dataset.
    """
    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        for i, qmd_feat in enumerate(['qmd1', 'qmd2', 'qmd3']):
            self.data[qmd_feat] = self.data['raw_qmd_features'][:,i]
            self.slices[qmd_feat] = self.slices['raw_qmd_features']  # TODO
        self.data['avg_qmd'] = torch.mean(self.data['raw_qmd_features'], dim=1).unsqueeze(1)
        self.slices['avg_qmd'] = self.slices['raw_qmd_features']
        if not isinstance(self.data, MolData):
            self.data = MolData.from_data(self.data)
            self.slices["atom_mol_batch"] = copy.copy(self.slices["Z"])
            self.slices["sample_id"] = copy.copy(self.slices["N"])

    @property
    def raw_file_names(self):  
        return ['qm_shift.pt']  # qm_shift.pt is the processed data for QMShiftDataset
    
    def download(self):  # no raw data provided to download
        pass
    
    @property
    def processed_file_names(self):
        return ['processed.pt']

    def get_duplicated_mols(self, save=False):
        """
        Get the list of duplicated molecules. Save the list if save=True.
        """
        duplicated_mols_path = osp.join(self.processed_dir, 'duplicated_mols.pt')
        if os.path.exists(duplicated_mols_path):
            return torch.load(duplicated_mols_path)
        
        qm_shift_data, qm_shift_slices = torch.load(self.raw_paths[0])
        mol_id = [i.item() for i in qm_shift_data['molecule_id']]
        duplicated_mol_id = [item for item, count in Counter(mol_id).items() if count > 1]
        print('duplicated molecule id number: ', len(duplicated_mol_id))  # 1401 for Carbon
        # for each duplicated molecule_id, find the list of moles.
        dup_moles = defaultdict(list)
        for idx, mol_id in enumerate(qm_shift_data['molecule_id']):
            if mol_id.item() in duplicated_mol_id:
                mask = qm_shift_data['mask'][qm_shift_slices['mask'][idx]:qm_shift_slices['mask'][idx+1]]
                atom_y = qm_shift_data['atom_y'][qm_shift_slices['atom_y'][idx]:qm_shift_slices['atom_y'][idx+1]]
                dup_moles[mol_id.item()].append({'mask': mask, 'atom_y': atom_y})
        if save:
            torch.save(dup_moles, duplicated_mols_path)
        return dup_moles

    def merge_duplicated_moles(self, save=False):
        """
        Merge the duplicated moles by keeping the maximum set of exp values for each molecule.
        e.g. [0,1,1,0,0], [0,1,0,0,0], [1,1,0,0,0] -> [1,1,1,0,0]
        """
        merged_mols_path = osp.join(self.processed_dir, 'merged_mols.pt')
        if os.path.exists(merged_mols_path):
            return torch.load(merged_mols_path)

        dup_moles = self.get_duplicated_mols()
        merged_moles = {}
        for mol_id, moles in dup_moles.items():
            merged_result = merge_duplicated_moles(moles)
            if merged_result is not None:
                merged_moles[mol_id] = merged_result
        if save:
            torch.save(merged_moles, merged_mols_path)
        return merged_moles

    def process(self):
        qm_shift_data, qm_shift_slices = torch.load(self.raw_paths[0])
        # create a new dataset with the merged moles and the rest of the moles
        data_list = []
        merged_moles = self.merge_duplicated_moles()  # 111 for Carbon
        added_merged_moles = []
        for idx, mol_id in enumerate(qm_shift_data['molecule_id']):
            mol_data = copy_mol_from_dataset(idx, qm_shift_data, qm_shift_slices)
            if mol_id.item() not in merged_moles:
                data_list.append(mol_data)
            elif mol_id.item() not in added_merged_moles:
                mol_data['mask'] = torch.tensor(merged_moles[mol_id.item()]['mask']).to(torch.float32)  # todo: dtype
                mol_data['atom_y'] = torch.tensor(merged_moles[mol_id.item()]['atom_y']).to(torch.float32)
                data_list.append(mol_data)
                added_merged_moles.append(mol_id.item())
            else:
                continue
        combine_mol_dict_to_dataset(data_list, self.processed_paths[0])