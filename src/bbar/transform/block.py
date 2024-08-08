from rdkit import Chem

from rdkit.Chem import Mol
from typing import Optional, List

from .base import MolGraphTransform
from .feature import get_atom_features, get_bond_features, get_bond_index

__all__ = ['BlockGraphTransform']

BRICS_label_list = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
BRICS_label_map = {label: idx for idx, label in enumerate(BRICS_label_list)}
NUM_BLOCK_FEATURES = len(BRICS_label_list)

def get_brics_label(brics_block: Mol, use_index: Optional[int] = 0) -> int:
    # Find BRICS Atom
    if use_index is None :
        brics_atom = None
        for atom in brics_block.GetAtoms() :
            if atom.GetAtomicNum() == 0 :
                brics_atom = atom
        assert brics_atom is not None
    else :
        brics_atom = brics_block.GetAtomWithIdx(0)
        assert brics_atom.GetAtomicNum() == 0

    brics_label = brics_atom.GetIsotope()
    return brics_label

def get_brics_feature(brics_block: Mol, use_index: Optional[int] = 0) -> List[float]:
    """
    Convert integer to One-Hot Vector (type: list)
    """
    brics_label = get_brics_label(brics_block, use_index)
    global_x = [0.] * NUM_BLOCK_FEATURES
    global_x[BRICS_label_map[brics_label]] = 1.
    return global_x

class BlockGraphTransform(MolGraphTransform) :
    atom_feature_fn  = staticmethod(get_atom_features)
    bond_feature_fn = staticmethod(get_bond_features)
    bond_index_fn = staticmethod(get_bond_index)
    mol_feature_fn = staticmethod(get_brics_feature)

