from typing import Callable, Optional
from torch_geometric.data import InMemoryDataset, Data
import torch
import pandas as pd
import os.path as osp
from utils.data_processing_utils import get_atom_y_and_mask, get_all_data_from_nmr_log, combine_mol_dict_to_dataset
from abc import ABC, abstractmethod
import os


class MolData(Data):
    """
    Data class for molecules. __inc__ is overwritten to handle the edge_index.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_data(data: Data):
        kwargs = {key: getattr(data, key) for key in data.keys}
        return MolData(**kwargs)

    def __inc__(self, key: str, value, *args, **kwargs):
        if "_edge_index" in key:
            return self.N
        return super(MolData, self).__inc__(key, value, *args, **kwargs)


class NMRDataset(InMemoryDataset, ABC):
    """
    Dataset class for molecules with both experiemental and DFT calculated nmr shift values.
    """
    def __init__(
        self,
        root: Optional[str] = None,
        nmr_log_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        """
        Args:
            root: root directory of the dataset.
            nmr_log_path: path containing all the nmr log files. This could also be a list of file path of nmr log files, depending on the implementation of self.get_nmr_log_path_list.
            transform, pre_transform, pre_filter: functions for data processing in the parent InMemoryDataset class. Suppose to be the abosolute path.
        """
        self.nmr_log_path = nmr_log_path
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if not isinstance(self.data, MolData):
            self.data = MolData.from_data(self.data)
        
    @property
    def raw_file_names(self):
        # download is disabled in this dataset because there's a huge number of log files
        return []

    @property
    def processed_file_names(self):
        return ['processed.pt']
    
    def screen_mol(self, mol_info):
        """
        Screen out the molecules according to certain conditions.
        return True if the molecule is to be included in the dataset, False otherwise.
        """
        return True
    
    @abstractmethod
    def get_shift_dict(self, mol_info):
        """
        Get the shift dict from the mol_info.
        """
        return {}

    @abstractmethod
    def get_mol_info_from_log(self, nmr_log_path):
        """
        Get mol_info from the nmr log file.
        return: dict containing the info of the molecule.
        """
        return {}
    
    @abstractmethod
    def get_nmr_log_path_list(self):
        """
        Get the list of nmr log file paths. These paths should be the absolute path.
        """
        return []

    def process(self):
        data_list = []
        self.nmr_log_path_list = self.get_nmr_log_path_list()
        for nmr_log_path in self.nmr_log_path_list:
            # read log
            mol_info = self.get_mol_info_from_log(nmr_log_path)
            if mol_info is None: continue
            # find the corresponding molecule exp shift value
            shift_dict = self.get_shift_dict(mol_info)
            atom_y, mask = get_atom_y_and_mask(shift_dict, mol_info['N'])
            # screen out the molecules according to certain conditions
            if not self.screen_mol(mol_info): continue
            # combine as a dict
            mol_info.update({'atom_y': atom_y, 'mask': mask})
            for attr in ['atom_y', 'R', 'raw_qmd_features', 'mask']:
                mol_info[attr] = torch.tensor(mol_info[attr]).to(torch.float32)
            for attr in ['Z', 'N']:
                mol_info[attr] = torch.tensor(mol_info[attr]).to(torch.long)
            data_list.append(mol_info)
        combine_mol_dict_to_dataset(data_list, self.processed_paths[0])


class NMRDatasetFromProcessed(NMRDataset):
    """
    Dataset class for already processed dataset, where the process function is not called.
    """
    def __init__(
        self,
        root: Optional[str] = None,
        nmr_log_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        super().__init__(root, nmr_log_path, transform, pre_transform, pre_filter)
    
    def process(self):
        pass
    
    def get_nmr_log_path_list(self):
        return []
    
    def get_shift_dict(self, mol_info):
        return {}
    
    def get_mol_info_from_log(self, nmr_log_path):
        return {}


class NMRDatasetFromCSV(NMRDataset):
    """
    Dataset class for molecules with both experiemental and DFT calculated nmr shift values. The nmr shift values are read from a csv file.
    Suppose the csv file is in the raw file path. The csv file should contain at least the following columns:
    - mol_id: id of the molecule, this should correspond to the name of the nmr log file. 
    - atom_type: type of the atom
    - atom_index: index of the atom
    - shift: nmr shift value
    """
    def __init__(
        self,
        root: Optional[str] = None,
        nmr_log_path: Optional[str] = None,
        csv_file_name: Optional[str] = 'nmr_shift.csv',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        self.csv_file = pd.read_csv(osp.join(root, 'raw', csv_file_name))
        self.csv_file_name = csv_file_name
        super().__init__(root, nmr_log_path, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return [self.csv_file_name]
    
    def get_mol_info_from_log(self, nmr_log_path):
        mol_info = get_all_data_from_nmr_log(nmr_log_path)
        # add name of the mol
        mol_info.update({'mol_id': self.get_mol_id_from_nmr_log_name(nmr_log_path)})
        return mol_info
    
    def get_shift_dict(self, mol_info):
        """
        Get the shift dict from the mol_info.
        """
        mol_csv_data = self.csv_file[self.csv_file['mol_id'] == mol_info['mol_id']]
        shift_dict = dict(zip(mol_csv_data['atom_index'], mol_csv_data['shift']))
        return shift_dict
    
    def get_mol_id_from_nmr_log_name(self, nmr_log_name):
        return osp.basename(nmr_log_name).split('.')[0]


class CaseStudyDataset(NMRDatasetFromCSV):
    """
    Dataset for the case study of the paper containing TIC-10 and NHP.
    Suppose all the nmr log files are in the raw file path under the same folder nmr_log_path.
    """
    def __init__(
            self, 
            root: Optional[str] = None,
            nmr_log_path: Optional[str] = 'NMR_output/',
            csv_file_name: Optional[str] = 'case_study_mols.csv',
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
    ):
        self.nmr_log_path = osp.join(root, 'raw', nmr_log_path)
        self.csv_file = pd.read_csv(osp.join(root, 'raw', csv_file_name))
        assert self.csv_file['atom_type'].nunique() == 1  # multiple atom types are currently not supported. Could be done by modifying self.csv_file here to screen out the atom type.
        self.csv_file_name = csv_file_name
        super(NMRDatasetFromCSV, self).__init__(root, self.nmr_log_path, transform, pre_transform, pre_filter)

    def get_nmr_log_path_list(self):
        # get the list of all the files under the path of nmr_log_path
        return [osp.join(self.nmr_log_path, f) for f in os.listdir(self.nmr_log_path) if osp.isfile(osp.join(self.nmr_log_path, f))]

