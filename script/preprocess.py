import argparse
import os
from typing import Optional
import parmap
import pickle
import pandas as pd
import random
from rdkit import Chem

from bbar.fragmentation import BRICS_BlockLibrary, brics_fragmentation


ATOM_SYMBOL = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I")


def convert_to_SMILES(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ATOM_SYMBOL:
            return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, help="Data Directory Path.")
    parser.add_argument("--cpus", type=int, default=1, help="Num Workers")
    parser.add_argument("--split_ratio", type=float, help="Train/Validation split ratio")
    parser.add_argument("--seed", default=1, help="random seed for split")
    args = parser.parse_args()
    random.seed(args.seed)

    mol_path = os.path.join(args.data_dir, "data.csv")
    valid_mol_path = os.path.join(args.data_dir, "valid_data.csv")
    split_path = os.path.join(args.data_dir, "split.csv")
    save_lib_path = os.path.join(args.data_dir, "library.csv")
    save_pkl_path = os.path.join(args.data_dir, "data.pkl")

    assert os.path.exists(mol_path)
    print("Load molecule")
    smiles_list = pd.read_csv(mol_path, usecols=["SMILES"])["SMILES"].to_list()
    entire_mol_list = [convert_to_SMILES(smiles) for smiles in smiles_list]
    mol_list = [mol for mol in entire_mol_list if mol is not None]
    valid_indices = [i for i, mol in enumerate(entire_mol_list) if mol is not None]
    print(f"{len(valid_indices)} molecules are remains except for unallowed atom types")

    print("Create Library")
    flag_list = BRICS_BlockLibrary.create_library_file(save_lib_path, mol_list, cpus=args.cpus)
    valid_indices = [idx for idx, flag in zip(valid_indices, flag_list, strict=True) if flag]
    mol_list = [mol for flag, mol in zip(flag_list, mol_list, strict=True) if flag]
    print(f"{len(valid_indices)} molecules are success to fragmentation")

    with open(mol_path) as f:
        lines = f.readlines()
        with open(valid_mol_path, "w") as w:
            w.write(lines[0])
            for idx in valid_indices:
                w.write(lines[1 + idx])

    print("Create Datapoints")
    fragmented_molecules = parmap.map(brics_fragmentation, mol_list, pm_processes=args.cpus, pm_pbar=True)
    with open(save_pkl_path, "wb") as f:
        pickle.dump(fragmented_molecules, f)

    print("Create Split")
    assert 0.0 < args.split_ratio < 1.0, "split ratio should be smaller than 1"
    indices = list(range(len(valid_indices)))
    random.shuffle(indices)
    num_train = int(len(indices) * args.split_ratio)
    with open(split_path, "w") as w:
        for i, idx in enumerate(indices):
            if i < num_train:
                w.write(f"train,{idx}\n")
            else:
                w.write(f"val,{idx}\n")
