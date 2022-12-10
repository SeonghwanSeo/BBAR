from rdkit import Chem

import torch
import numpy as np
from torch.utils.data import Dataset

from tqdm import tqdm
import parmap

from typing import Optional, List, Dict, Union, Tuple
from torch import BoolTensor, FloatTensor
from torch_geometric.data import Data as PyGData
from rdkit.Chem import Mol
from bbar.utils.typing import SMILES

from bbar.fragmentation import BRICS_FragmentedGraph, BRICS_Fragmentation, BRICS_BlockLibrary
from bbar.transform import CoreGraphTransform

class BBARDataset(Dataset) :
    def __init__(self,
        molecules: List[Union[Mol, SMILES]],
        fragmented_molecules: Optional[List[BRICS_FragmentedGraph]],
        properties: List[Dict[str, float]],
        library: BRICS_BlockLibrary,
        library_pygdata_list: List[PyGData],
        library_frequency: FloatTensor,
        num_negative_samples: int,
        train: bool,
    ) :
        super(BBARDataset, self).__init__()
        
        assert len(molecules) == len(properties)

        self.molecules = molecules
        self.properties = properties
        self.library = library
        self.library_pygdata = library_pygdata_list
        self.library_frequency = library_frequency
        self.num_negative_samples = num_negative_samples
        self.train = train

        self.core_transform = CoreGraphTransform.call 

        if fragmented_molecules is None :
            fragmentation: BRICS_Fragmentation = self.library.fragmentation
            self.fragmented_molecules = [fragmentation(mol) for mol in tqdm(molecules)]
        else :
            assert len(molecules) == len(fragmented_molecules)
            self.fragmented_molecules = fragmented_molecules 

    def __len__(self) -> int:
        return len(self.fragmented_molecules)

    def __getitem__(self, idx):
        # Load Datapoint
        fragmented_mol = self.fragmented_molecules[idx]
        core_rdmol, block_idx, core_atom_idx = self.get_datapoint(fragmented_mol)
        pygdata_core = self.core_transform(core_rdmol)

        # Load Condition
        condition: Dict[str, float] = self.properties[idx]

        # Set Answer
        num_core_atoms = core_rdmol.GetNumAtoms()
        if block_idx == None :
            y_term: bool = True
            y_atom: BoolTensor = torch.full((num_core_atoms,), False, dtype=torch.bool)
        else :
            y_term: bool = False
            y_atom: BoolTensor = torch.full((num_core_atoms,), False, dtype=torch.bool)
            y_atom[core_atom_idx] = True

        pygdata_core.y_term = y_term
        pygdata_core.y_atom = y_atom

        if self.train :
            pos_pygdata: PyGData = None
            neg_pygdatas: List[PyGData] = None
            if block_idx == None :
                pos_pygdata = self.library_pygdata[0]
                neg_pygdatas = [self.library_pygdata[0]] * self.num_negative_samples
            else :
                pos_pygdata = self.library_pygdata[block_idx]
                neg_idxs = self.get_negative_samples(block_idx)
                neg_pygdatas = [self.library_pygdata[idx] for idx in neg_idxs]
            return pygdata_core, condition, pos_pygdata, *neg_pygdatas
        else :
            if block_idx == None :
                pos_idx = 0
                neg_idxs = [0] * self.num_negative_samples
            else :
                pos_idx = block_idx
                neg_idxs = self.get_negative_samples(block_idx)
            return pygdata_core, condition, pos_idx, *neg_idxs 


    def get_datapoint(self, fragmented_mol) :
        datapoint = fragmented_mol.get_datapoint()
        core_rdmol, block_rdmol, (core_atom_idx, _) = datapoint
        if block_rdmol is not None :
            block_idx = self.library.get_index(block_rdmol)
            datapoint = (core_rdmol, block_idx, core_atom_idx)
        else :
            datapoint = (core_rdmol, None, None)

        return datapoint

    def get_negative_samples(self, positive_sample: int) -> List[int]:
        freq = torch.clone(self.library_frequency)
        freq[positive_sample] = 0.0
        neg_idxs = torch.multinomial(freq, self.num_negative_samples, True).tolist()
        return neg_idxs
