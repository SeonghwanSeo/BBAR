from rdkit import Chem
from rdkit.Chem import Mol, Atom, BondType

from typing import Optional


# For pyrrole, we should change Explicit H  ex) c1ccc(-n2cccc2)cc1 <-> c1ccccc1 + c1cc[nH]c1
def create_bond(rwmol, idx1, idx2, bondtype):
    rwmol.AddBond(idx1, idx2, bondtype)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        atom_numexplicitHs = atom.GetNumExplicitHs()
        if atom_numexplicitHs:
            atom.SetNumExplicitHs(atom_numexplicitHs - 1)


def remove_bond(rwmol, idx1, idx2):
    rwmol.RemoveBond(idx1, idx2)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "N" and atom.GetIsAromatic() is True:
            atom.SetNumExplicitHs(1)


def check_dummy_atom(atom) -> bool:
    return atom.GetAtomicNum() == 0


def add_dummy_atom(rwmol, index, bondtype=BondType.SINGLE, label=0):
    dummy_atom = Atom("*")
    dummy_atom.SetIsotope(label)  # default: 0
    new_idx = rwmol.AddAtom(dummy_atom)
    create_bond(rwmol, index, new_idx, bondtype)


def find_dummy_atom(rdmol) -> Optional[int]:
    for idx, atom in enumerate(rdmol.GetAtoms()):
        if check_dummy_atom(atom):
            return idx
    return None


def get_dummy_bondtype(dummy_atom) -> BondType:
    bondtype = dummy_atom.GetTotalValence()
    assert bondtype in [1, 2, 3]
    if bondtype == 1:
        return BondType.SINGLE
    elif bondtype == 2:
        return BondType.DOUBLE
    else:
        return BondType.TRIPLE


def create_monoatomic_mol(symbol: str) -> Mol:
    return Chem.MolFromSmiles(symbol)


def merge(core: Mol, block: Mol, index1, index2=None) -> Mol:
    """
    connect index'th atom of scaffold and fragment.
    ex) NC1CCCCC1, *=C(C)C, 3, None -> NC1CC(=C(C)C)CCC1
    ex) c1ccccc1, *N(C)C, 0, None -> CN(C)c1ccccc1
    """
    if index2 is None:
        index2 = find_dummy_atom(fragment)
    assert index2 is not None

    rwmol = Chem.RWMol(Chem.CombineMols(scaffold, fragment))
    dummy_atom_index = scaffold.GetNumAtoms() + index2

    atom1 = rwmol.GetAtomWithIdx(index1)
    dummy_atom = rwmol.GetAtomWithIdx(dummy_atom_index)
    assert check_dummy_atom(dummy_atom)
    bondtype = get_dummy_bondtype(dummy_atom)

    index2 = dummy_atom.GetNeighbors()[0].GetIdx()
    create_bond(rwmol, index1, index2, bondtype)
    rwmol.RemoveAtom(dummy_atom_index)
    mol = rwmol.GetMol()
    Chem.SanitizeMol(mol)

    return mol
