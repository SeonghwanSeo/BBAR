import torch

from rdkit.Chem import Mol, Bond
from torch_geometric.typing import Adj
from typing import List, Tuple

def get_bonds_normal(mol: Mol) -> Tuple[List[Bond], Adj]:
    bonds = list(mol.GetBonds())
    if len(bonds) > 0 :
        bond_index = torch.LongTensor([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in bonds]).T
        bond_index = torch.cat([bond_index, bond_index[[1,0]]], dim=1)          # (2, E), E = NumRealBonds
        bonds = bonds * 2
        return bonds, bond_index
    else :
        return [], torch.zeros((2,0), dtype=torch.long)

# UNUSED
def get_bonds_complete(mol: Mol) -> Tuple[List[Bond], Adj]:
    num_atoms = mol.GetNumAtoms()
    if num_atoms > 1 :
        bond_index = [[i, j] for i, j in zip(range(num_atoms), range(num_atoms)) if i != j]
        bonds = [mol.GetBondBetweenAtoms(i, j) for i, j in bond_index]          # (None for virtual(unreal) edge)
        bond_index = torch.LongTensor(bond_index).T                             # (2, E), E = V(V-1)
        return bonds, bond_index
    else :
        return [], torch.zeros((2,0), dtype=torch.long)
