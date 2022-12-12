from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# rdkit Descriptors
rdmol_desc_list = {
    'mw': Descriptors.ExactMolWt,
    'logp': Descriptors.MolLogP,
    'tpsa': Descriptors.TPSA,
    'qed': Descriptors.qed,
}

property_list = ['mw', 'tpsa', 'logp', 'qed']
floating_point = {
    'mw': 2,
    'tpsa': 3,
    'logp': 5,
    'qed': 5,
}

with open('./smiles/train.smi') as f :
    train_lines = f.readlines()

with open('./smiles/val.smi') as f :
    val_lines = f.readlines()

with open('./smiles/test.smi') as f :
    test_lines = f.readlines()


data_writer = open('./data.csv', 'w') 
split_writer = open('./split.csv', 'w')

data_writer.write('SMILES,' + ','.join(property_list) + '\n')
idx = 0
for l in tqdm(train_lines) :
    smiles = l.strip()
    properties = []
    rdmol = Chem.MolFromSmiles(smiles)
    for key in property_list :
        value = rdmol_desc_list[key](rdmol)
        properties.append(f'{value:.{floating_point[key]}f}')
    smiles = Chem.MolToSmiles(rdmol, isomericSmiles=False)
    data_writer.write(f'{smiles},{",".join(properties)}\n')
    split_writer.write(f'train,{idx}\n')
    idx += 1

for l in tqdm(val_lines) :
    smiles = l.strip()
    properties = []
    rdmol = Chem.MolFromSmiles(smiles)
    for key in property_list :
        value = rdmol_desc_list[key](rdmol)
        properties.append(f'{value:.{floating_point[key]}f}')
    smiles = Chem.MolToSmiles(rdmol, isomericSmiles=False)
    data_writer.write(f'{smiles},{",".join(properties)}\n')
    split_writer.write(f'val,{idx}\n')
    idx += 1

for l in tqdm(test_lines) :
    smiles = l.strip()
    properties = []
    rdmol = Chem.MolFromSmiles(smiles)
    for key in property_list :
        value = rdmol_desc_list[key](rdmol)
        properties.append(f'{value:.{floating_point[key]}f}')
    smiles = Chem.MolToSmiles(rdmol, isomericSmiles=False)
    data_writer.write(f'{smiles},{",".join(properties)}\n')
    split_writer.write(f'test,{idx}\n')
    idx += 1

data_writer.close()
split_writer.close()
