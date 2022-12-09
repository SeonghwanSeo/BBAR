from rdkit import Chem

import torch
import numpy as np
from torch import BoolTensor
from torch.utils.data import Dataset

from tqdm import tqdm
import parmap

from typing import Optional, List, Dict, Union, Tuple
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
        cache_ratio: float = 0.0,
    ) :
        super(BBARDataset, self).__init__()
        
        assert len(molecules) == len(properties)
        assert cache_ratio >= 0.0 and cache_ratio <= 1.0

        self.molecules = molecules
        self.properties = properties
        self.library = library
        self.core_transform = CoreGraphTransform.call 
        self.cache_ratio = cache_ratio

        if fragmented_molecules is None :
            fragmentation: BRICS_Fragmentation = self.library.fragmentation
            self.fragmented_molecules = [fragmentation(mol) for mol in tqdm(molecules)]
        else :
            assert len(molecules) == len(fragmented_molecules)
            self.fragmented_molecules = fragmented_molecules 
        self.datapoints: List[[Tuple[Mol, int, int]]] = None 
        """
        datapoint: Tuple[ RDMol-Core, LibraryIndex-Block, AtomIdx-Core ]
        """

    def __len__(self) -> int:
        return len(self.fragmented_molecules)

    def __getitem__(self, idx):
        # Load Datapoint
        fragmented_mol = self.fragmented_molecules[idx]
        core_rdmol, block_idx, core_atom_idx = self.get_datapoint(fragmented_mol)

        pygdata = self.core_transform(core_rdmol)

        # Load Condition
        condition: Dict[str, float] = self.properties[idx]

        # Set Answer
        num_core_atoms = core_rdmol.GetNumAtoms()
        if block_idx == None :
            y_term: bool = True
            y_block: int = 0            # Dummy value. Masked during Training.
            y_atom: BoolTensor = torch.full((num_core_atoms,), False, dtype=torch.bool)
        else :
            y_term: bool = False
            y_block: int = block_idx 
            y_atom: BoolTensor = torch.full((num_core_atoms,), False, dtype=torch.bool)
            y_atom[core_atom_idx] = True

        pygdata.y_term = y_term
        pygdata.y_block = y_block
        pygdata.y_atom = y_atom
        return pygdata, condition

    def get_datapoint(self, fragmented_mol) :
        datapoint = fragmented_mol.get_datapoint()
        core_rdmol, block_rdmol, (core_atom_idx, _) = datapoint
        if block_rdmol is not None :
            block_idx = self.library.get_index(block_rdmol)
            datapoint = (core_rdmol, block_idx, core_atom_idx)
        else :
            datapoint = (core_rdmol, None, None)
        return datapoint
