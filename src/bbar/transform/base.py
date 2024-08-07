import torch

from rdkit import Chem

from typing import Optional, Union, Dict, Tuple, List, Callable
from torch import FloatTensor
from torch_geometric.data import Data as PyGData
from torch_geometric.typing import Adj
from rdkit.Chem import Atom, Bond, Mol
from bbar.utils.typing import NodeVector, EdgeVector, GlobalVector, SMILES

from bbar.utils.common import check_and_convert_to_rdmol
from .feature import get_atom_features, get_bond_features, get_bond_index, get_mol_features

__all__ = ['MolGraphTransform']

class MolGraphTransform() :
    atom_feature_fn: Callable[[Atom], List[float]] = staticmethod(get_atom_features)
    bond_feature_fn: Optional[Callable[[Bond], List[float]]] = staticmethod(get_bond_features)
    bond_index_fn: Optional[Callable[[Mol], Tuple[List[Bond], Adj]]] = staticmethod(get_bond_index)
    mol_feature_fn: Optional[Callable[[Mol], List[float]]] = staticmethod(get_mol_features)

    def __call__(self, mol: Union[SMILES, Mol]) -> PyGData:
        return self.call(mol) 

    @classmethod
    def call(cls, mol: Union[SMILES, Mol]) -> PyGData:
        return PyGData(**cls.processing(mol))

    @classmethod
    def processing(cls, mol: Union[SMILES, Mol]) -> Dict:
        rdmol = check_and_convert_to_rdmol(mol)

        retval = {}

        # Get Atom Feature
        retval['x'] = cls.get_atom_feature(rdmol)

        # Get Bond Index & Type
        if cls.bond_index_fn is not None :
            bonds, bond_index = cls.get_bond_index(rdmol)
        else :
            bonds, bond_index = [], torch.zeros((2,0), dtype=torch.long)
        retval['edge_index'] = bond_index

        if cls.bond_feature_fn is not None :
            retval['edge_attr'] = cls.get_bond_feature(bonds)

        if cls.mol_feature_fn is not None :
            retval['global_x'] = cls.get_mol_feature(rdmol)

        return retval
    
    @classmethod
    def get_atom_feature(cls, rdmol: Mol) -> NodeVector :          # (V, Fv)
        return torch.FloatTensor([cls.atom_feature_fn(atom) for atom in rdmol.GetAtoms()])

    @classmethod
    def get_bond_index(cls, rdmol: Mol) -> Tuple[List[Bond], Adj] :
        return cls.bond_index_fn(rdmol)

    @classmethod
    def get_bond_feature(cls, bonds: List[Bond]) -> EdgeVector :   # (E, Fe)
        return torch.FloatTensor([cls.bond_feature_fn(bond) for bond in bonds])
    
    @classmethod
    def get_mol_feature(cls, rdmol: Mol) -> GlobalVector :         # (1, Fg)
        return torch.FloatTensor(cls.mol_feature_fn(rdmol)).unsqueeze(0)

