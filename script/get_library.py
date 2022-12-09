from rdkit import Chem
from rdkit.Chem import Mol
import argparse
import os

import sys
sys.path.append(".")
sys.path.append("..")

from typing import List
from bbar.utils.typing import SMILES

from bbar.fragmentation import BRICS_BlockLibrary

def load_mol_file(mol_path: str) -> List[SMILES]:
    extension = os.path.splitext(mol_path)[1]
    assert extension in ['.smi', '.csv'], 'The extension of mol_path should be `.csv` or `.smi`'
    if extension == '.smi' :
        with open(mol_path) as f :
            smiles_list = [l.strip() for l in f.readlines()]
    else :
        import pandas as pd
        smiles_list = pd.read_csv(mol_path, usecols=['SMILES'])['SMILES'].to_list()

    smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False) for smiles in smiles_list]

    return smiles_list 

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--mol_path', type=str, help='SMILES File Path.')
    parser.add_argument('--out_path', type=str, help='Save Path for Library, Extension should be .smi or .csv')
    parser.add_argument('--cpus', type=int, default=1)
    args = parser.parse_args()
    
    mol_list = load_mol_file(args.mol_path)
    BRICS_BlockLibrary.create_library_file(args.out_path, mol_list, cpus=args.cpus)
