from collections import OrderedDict

from rdkit import Chem
from rdkit.Chem import Atom, Mol

__all__ = ['get_atom_features', 'NUM_ATOM_FEATURES']

ATOM_SYMBOL = ('*', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I')
DEGREE = (0, 1, 2, 3, 4, 5, 6)
VALENCE = (0, 1, 2, 3, 4, 5, 6)
FORMAL_CHARGE = (-1, 0, 1)
NUM_HS = (0, 1, 2, 3, 4)
HYBRIDIZATION = (
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
EN = {
    '*' : 0.00,
    'C' : 2.55,
    'N' : 3.04,
    'O' : 3.44,
    'F' : 3.98,
    'P' : 2.19,
    'S' : 2.59,
    'Cl': 3.16,
    'Br': 2.96,
    'I' : 2.66,
}

FEATURE_INFORM = OrderedDict([
    ['symbol', {'choices' : ATOM_SYMBOL, 'allow_unknown' : False}],
    ['degree', {'choices' : DEGREE, 'allow_unknown' : True}],
    ['valence', {'choices' : VALENCE, 'allow_unknown' : True}],
    ['formal_charge', {'choices' : FORMAL_CHARGE, 'allow_unknown' : True}],
    ['num_Hs', {'choices' : NUM_HS, 'allow_unknown' : True}],
    ['hybridization', {'choices' : HYBRIDIZATION, 'allow_unknown' : True}],
    ['aromatic', {'choices' : None}],
    ['mass', {'choices': None}],
    ['EN', {'choices': None}],
])

for key, val in FEATURE_INFORM.items() :
    if val['choices'] is None :
        val['dim'] = 1
    else :
        val['choices'] = {v: i for i, v in enumerate(val['choices'])}
        if val['allow_unknown'] :
            val['dim'] = len(val['choices']) + 1
        else :
            val['dim'] = len(val['choices'])
 
NUM_KEYS = len(FEATURE_INFORM)
NUM_ATOM_FEATURES = sum([val['dim'] for val in FEATURE_INFORM.values()])

def get_atom_features(atom: Atom) :
    symbol = atom.GetSymbol()
    features = {
        'symbol' : symbol,
        'degree' : atom.GetTotalDegree(),
        'valence' : atom.GetTotalValence(),
        'formal_charge' : atom.GetFormalCharge(),
        'num_Hs' : atom.GetTotalNumHs(),
        'hybridization' : atom.GetHybridization(),
        'aromatic' : atom.GetIsAromatic(),        # True of False
        'mass' : atom.GetMass() * 0.01,           # scaling
        'EN' : EN[symbol] * 0.25,                 # scaling
    }
    return _get_sparse(features)
   
def _get_sparse(features: dict) -> list :
    retval = [0] * NUM_ATOM_FEATURES
    idx = 0
    for key, inform in FEATURE_INFORM.items() :
        choices, dim = inform['choices'], inform['dim']
        x = features[key]
        if choices is None :
            retval[idx] = x
        elif inform['allow_unknown'] is True :
            retval[idx + choices.get(x, dim - 1)] = 1
        else :
            retval[idx + choices[x]] = 1
        idx += dim
    return retval
