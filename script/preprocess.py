from rdkit import Chem
from rdkit.Chem import Mol
import argparse
import os
import parmap
import pickle

import sys
sys.path.append(".")
sys.path.append("..")

from typing import List
from bbar.utils.typing import SMILES

from bbar.fragmentation import brics_fragmentation
from bbar.utils.common import convert_to_SMILES

def load_mol_file(mol_path: str) -> List[SMILES]:
    extension = os.path.splitext(mol_path)[1]
    assert extension in ['.smi', '.csv'], 'The extension of mol_path should be `.csv` or `.smi`'
    if extension == '.smi' :
        with open(mol_path) as f :
            smiles_list = [l.strip() for l in f.readlines()]
    else :
        import pandas as pd
        smiles_list = pd.read_csv(mol_path, usecols=['SMILES'])['SMILES'].to_list()

    smiles_list = [convert_to_SMILES(smiles, isomericSmiles=False) for smiles in smiles_list]

    return smiles_list 

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Data Directory Path.')
    parser.add_argument('--cpus', type=int, default=1)
    args = parser.parse_args()

    mol_path = os.path.join(args.data_dir, 'data.csv')
    out_path = os.path.join(args.data_dir, 'data.pkl')

    mol_list = load_mol_file(mol_path)
    fragmented_molecules = parmap.map(brics_fragmentation, mol_list, pm_processes=args.cpus, pm_pbar=True)
    with open(out_path, 'wb') as f :
        pickle.dump(fragmented_molecules, f)
