from torch_geometric.data import InMemoryDataset, Data
from nmr_dataset import MolData
import torch
from typing import Optional
import pandas as pd
from utils.data_processing_utils import read_combined_file, get_atom_y_and_mask, get_all_data_from_nmr_log, combine_mol_dict_to_dataset
import copy


def read_combined_file(file_path, divider='--Link1--'):
    """
    read the combined file (like .com or .log), which have multiple molecules in one file divided by dividers like '--Link1--'
    return a list of blocks, each block is a list of lines
    """
    com_blocks = []
    with open(file_path, 'r') as com_file:
        block = []
        for line in com_file:
            if divider in line:
                if block:
                    com_blocks.append(block)
                    block = []  # Reset the block
            block.append(line)
        if block:
            com_blocks.append(block)
    return com_blocks


class CheshireDataset(InMemoryDataset):
    """
    Probe dataset for Cheshire dataset.
    """
    def __init__(self, root: Optional[str] = None, type: str = 'C', **kwargs):
        self.type = type
        assert self.type in ['C', 'H']
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0]) if type == 'C' else torch.load(self.processed_paths[1])
        self.data['avg_qmd'] = torch.mean(self.data['raw_qmd_features'], dim=1).unsqueeze(1)
        self.slices['avg_qmd'] = self.slices['raw_qmd_features']
        if not isinstance(self.data, MolData):
            self.data = MolData.from_data(self.data)
            self.slices["atom_mol_batch"] = copy.copy(self.slices["Z"])
            self.slices["sample_id"] = copy.copy(self.slices["N"])
    
    @property
    def raw_file_names(self):
        return ['CHESHIRE_probe.csv', 'ProbeSetNMR.log']
    
    @property
    def processed_file_names(self):
        return ['processed_c.pt', 'processed_h.pt']

    def download(self):
        pass

    def process(self):
        nmr_logs = read_combined_file(self.raw_paths[1], divider='Normal termination')[:-1]  # remove the last one which is only the line of "Normal termination"
        csv = pd.read_csv(self.raw_paths[0])
        current_processed_path = self.processed_paths[0] if self.type == 'C' else self.processed_paths[1]
        atom_type = 6 if self.type == 'C' else 1
        data_list = []
        for mol_idx, nmr_log in enumerate(nmr_logs):
            if mol_idx == 22: continue  # skip the charged one
            mol_info = get_all_data_from_nmr_log(nmr_log)
            assert mol_info is not None

            # get atom_y and mask
            mol_csv_data = csv[csv['mol_id'] == mol_idx]
            mol_csv_data = mol_csv_data[mol_csv_data['atom_type'] == atom_type]
            shift = mol_csv_data['Shift']
            atom_index_with_shift = mol_csv_data['atom_index']
            shift_dict = dict(zip(atom_index_with_shift, shift))
            atom_y, mask = get_atom_y_and_mask(shift_dict, mol_info['N'])
            
            # pack the info together in torch format into a dict
            mol_info.update({'atom_y': atom_y, 'mask': mask, 'mol_idx': mol_idx})
            for attr in ['atom_y', 'R', 'raw_qmd_features', 'mask']:
                mol_info[attr] = torch.tensor(mol_info[attr]).to(torch.float32)
            for attr in ['Z', 'N', 'mol_idx']:
                mol_info[attr] = torch.tensor(mol_info[attr]).to(torch.long)
            data_list.append(mol_info)
        combine_mol_dict_to_dataset(data_list, current_processed_path)
