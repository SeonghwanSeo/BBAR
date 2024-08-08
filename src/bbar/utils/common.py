from rdkit import Chem

from typing import Union
from .typing import SMILES
from rdkit.Chem import Mol

def convert_to_rdmol(mol: Union[SMILES, Mol], isomericSmiles: bool = True) -> Mol:
    if isomericSmiles :
        if isinstance(mol, SMILES) :
            mol = Chem.MolFromSmiles(mol)
    else :
        smiles = convert_to_SMILES(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smiles)
    return mol

def convert_to_SMILES(mol: Union[SMILES, Mol], canonicalize: bool = False,
                      isomericSmiles: bool = True) -> SMILES:
    if isomericSmiles :
        if isinstance(mol, Mol) :
            mol = Chem.MolToSmiles(mol)
        elif canonicalize is True :
            mol = Chem.MolToSmiles(Chem.MolFromSmiles(mol))
    else :
        if isinstance(mol, Mol) :
            mol = Chem.MolToSmiles(mol, isomericSmiles=False)
        else :
            mol = Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=False)
    return mol

def check_and_convert_to_rdmol(mol: Union[SMILES, Mol], isomericSmiles = True) -> Mol:
    assert isinstance(mol, Mol) or isinstance(mol, SMILES)
    return convert_to_rdmol(mol, isomericSmiles)

def check_and_convert_to_SMILES(mol: Union[SMILES, Mol], canonicalize: bool = False,
                                isomericSmiles = True) -> SMILES:
    assert isinstance(mol, Mol) or isinstance(mol, SMILES)
    return convert_to_SMILES(mol, canonicalize, isomericSmiles)

