from . import _atom, _bond, _edge_index
from torch_geometric.typing import Adj

NUM_ATOM_FEATURES = _atom.NUM_ATOM_FEATURES
NUM_BOND_FEATURES = _bond.NUM_BOND_FEATURES
   
get_atom_features = _atom.get_atom_features
get_bond_features = _bond.get_bond_features
get_bond_index = _edge_index.get_bonds_normal
get_mol_features = lambda mol: []
