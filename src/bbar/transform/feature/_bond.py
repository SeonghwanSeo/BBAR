from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from typing import Optional, List 
from rdkit.Chem import Bond

__all__ = ['get_bond_features', 'NUM_BOND_FEATURES']

### Define RDKit BOND TYPES ###
RDKIT_BOND_TYPES = {
    BondType.SINGLE: 0,
    BondType.DOUBLE: 1,
    BondType.TRIPLE: 2,
    BondType.AROMATIC: 3,
    BondType.OTHER: 4,
}
NUM_BOND_FEATURES = len(RDKIT_BOND_TYPES)

def get_bond_features(bond: Optional[Bond]) -> List[int] :
    retval = [0, 0, 0, 0, 0]
    bt = RDKIT_BOND_TYPES.get(bond.GetBondType(), 4)
    retval[bt] = 1
    return retval
