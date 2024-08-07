from .base import MolGraphTransform
from .feature import get_atom_features, get_bond_features, get_bond_index

class CoreGraphTransform(MolGraphTransform) :
    atom_feature_fn = staticmethod(get_atom_features)
    bond_feature_fn = staticmethod(get_bond_features)
    bond_index_fn = staticmethod(get_bond_index)
    mol_feature_fn = None
