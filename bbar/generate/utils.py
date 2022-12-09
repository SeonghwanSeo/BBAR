import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol, Atom
from rdkit.Chem import rdChemReactions as Reactions
from typing import Union, List, Tuple, Optional, Dict
import re
import pandas as pd
import gc
import logging

from bbar.utils.typing import SMILES
from bbar.utils.common import convert_to_SMILES, convert_to_rdmol

p = re.compile('\[\d+\*\]')

BRICS_ENV = {
    '1': ['3', '5', '10'],
    '3': ['1', '4', '13', '14', '15', '16'],
    '4': ['3', '5', '11'],
    '5': ['1', '4', '12', '13', '14', '15', '16'],
    '6': ['13', '14', '15', '16'],
    '7': ['7'],
    '8': ['9', '10', '13', '14', '15', '16'],
    '9': ['8', '13', '14', '15', '16'],
    '10': ['1', '8', '13', '14', '15', '16'],
    '11': ['4', '13', '14', '15', '16'],
    '12': ['5'],
    '13': ['3', '5', '6', '8', '9', '10', '11', '14', '15', '16'],
    '14': ['3', '5', '6', '8', '9', '10', '11', '13', '14', '15', '16'],
    '15': ['3', '5', '6', '8', '9', '10', '11', '13', '14', '16'],
    '16': ['3', '5', '6', '8', '9', '10', '11', '13', '14', '15', '16'],
}
BRICS_ENV_INT = {k: [int(_) for _ in v] for k, v in BRICS_ENV.items()}

BRICS_SMARTS_FRAG = {
  '1': ('[C;D3]([#0,#6,#7,#8])(=O)', '-'),
  '3': ('[O;D2]-;!@[#0,#6,#1]', '-'), 
  '4': ('[C;!D1;!$(C=*)]-;!@[#6]', '-'), 
  '5': ('[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]', '-'),
  '6': ('[C;D3;!R](=O)-;!@[#0,#6,#7,#8]', '-'), 
  '7': ('[C;D2,D3]-[#6]', '='),
  '8': ('[C;!R;!D1;!$(C!-*)]', '-'), 
  '9': ('[n;+0;$(n(:[c,n,o,s]):[c,n,o,s])]', '-'),
  '10': ('[N;R;$(N(@C(=O))@[C,N,O,S])]', '-'), 
  '11': ('[S;D2](-;!@[#0,#6])', '-'), 
  '12': ('[S;D4]([#6,#0])(=O)(=O)', '-'), 
  '13': ('[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]', '-'), 
  '14': ('[c;$(c(:[c,n,o,s]):[n,o,s])]', '-'),
  '15': ('[C;$(C(-;@C)-;@C)]', '-'), 
  '16': ('[c;$(c(:c):c)]', '-'),
}

BRICS_SMARTS_MOL = {
  '1': ('[C;D2;!H0]([#0,#6,#7,#8])(=O)', '-'),
  '3': ('[O;!H0]-;!@[#0,#6,#1]', '-'), 
  '4': ('[C;!H0;!$(C=*)]-;!@[#6]', '-'), 
  '5': ('[N;!H0;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]', '-'),
  '6': ('[C;D2;!H0;!R](=O)-;!@[#0,#6,#7,#8]', '-'), 
  '7': ('[C;D1,D2;H2,H3]-[#6]', '='),
  '8': ('[C;!H0;!R;!$(C!-*)]', '-'), 
  '9': ('[n;+0;!H0;$(n(:[c,n,o,s]):[c,n,o,s])]', '-'),
  '10': ('[N;!H0;R;$(N(@C(=O))@[C,N,O,S])]', '-'), 
  '11': ('[S;D1;!H0](-;!@[#0,#6])', '-'), 
  '12': ('[S;D3;!H0]([#6,#0])(=O)(=O)', '-'), 
  '13': ('[C;!H0;$(C(-;@[C,N,O,S])-;@[N,O,S])]', '-'), 
  '14': ('[c;!H0;$(c(:[c,n,o,s]):[n,o,s])]', '-'),
  '15': ('[C;!H0;$(C(-;@C)-;@C)]', '-'), 
  '16': ('[c;!H0;$(c(:c):c)]', '-'),
}

buildTemplate= []
buildReaction = []
for typ1, typ2list in BRICS_ENV.items() :
    for typ2 in typ2list :
        r1 = BRICS_SMARTS_MOL[typ1][0]
        r2, bond = BRICS_SMARTS_FRAG[typ2]
        react = '[$(%s):1].[$(%s):2]%s;!@[%s*]' % (r1, r2, bond, typ2)
        prod = '[*:1]%s;!@[*:2]' % (bond)
        tmpl = '%s>>%s' % (react, prod)
        buildTemplate.append(tmpl)
buildReaction = [Reactions.ReactionFromSmarts(template) for template in buildTemplate]

"""
composing style
Molecule + Block(With Star) -> Molecule
"""

BRICS_substructure = {k:Chem.MolFromSmarts(v[0]) for k, v in BRICS_SMARTS_MOL.items()}

def compose(
    core: Union[SMILES, Mol],
    block: Union[SMILES, Mol],
    atom_idx_core: int,
    atom_idx_block: int,
    returnMol: bool = True,
    ) -> Union[SMILES, Mol] :

    core = convert_to_rdmol(core)
    block = convert_to_rdmol(block)

    # Validity Check
    atom_core = core.GetAtomWithIdx(atom_idx_core)
    atom_block = block.GetAtomWithIdx(atom_idx_block)
    if (atom_block.GetAtomicNum() != 0):
        logging.debug(f"Block's {atom_idx_block}th atom '{atom_block.GetSymbol()}' should be [*].")
        return None

    brics_label_block = str(atom_block.GetIsotope())

    validity = False
    for brics_label in BRICS_ENV[brics_label_block] :
        substructure = BRICS_substructure[brics_label]
        for idxs_list in core.GetSubstructMatches(substructure) :
            if atom_idx_core == idxs_list[0] :
                validity = True
                break
    if not validity :
        logging.debug(f"Core's {atom_idx_core}th atom '{atom_core.GetSymbol()}' couldn't be connected with block.")
        return None

    # Combine Molecules
    num_atoms_core = core.GetNumAtoms()
    neigh_atom_idx_block = atom_block.GetNeighbors()[0].GetIdx()
    bondtype = Chem.rdchem.BondType.SINGLE if brics_label_block != '7' else Chem.rdchem.BondType.DOUBLE

    edit_mol = Chem.RWMol(Chem.CombineMols(core, block))
    atom_mol = edit_mol.GetAtomWithIdx(atom_idx_core)

    atom_numexplicitHs = atom_mol.GetNumExplicitHs()
    if atom_numexplicitHs :
        atom_mol.SetNumExplicitHs(atom_numexplicitHs - 1)

    edit_mol.AddBond(atom_idx_core,
                     num_atoms_core + neigh_atom_idx_block,
                     order = bondtype)
    edit_mol.RemoveAtom(num_atoms_core + atom_idx_block)   # Remove Dummy Atom

    combined_mol = edit_mol.GetMol()

    if returnMol :
        Chem.SanitizeMol(combined_mol)
    else :
        combined_mol = convert_to_SMILES(combined_mol)

    return combined_mol

def get_possible_indexs(core: Union[SMILES, Mol],
                        block: Union[SMILES, Mol, None] = None) -> List[Tuple[int, str]] :
    """
    Get Indexs which can be connected to target brics type for target blockment
    Return List[Tuple(AtomIndex:int, BRICSIndex:str)]
    Example
    >>> smiles_core = 'C(=O)CCNC=O'
    >>> smiles_block = '[10*]N1C(=O)COC1=O'
    >>> get_possible_indexs(smiles_core, smiles_block)
    [(0, '1'), (5, '1'), (2, '8'), (3, '8')]
    """
    core = convert_to_rdmol(core)

    if block is not None :
        block = convert_to_rdmol(block)
        brics_label_block = str(block.GetAtomWithIdx(0).GetIsotope())
        brics_label_list = BRICS_ENV[brics_label_block]
    else :
        brics_label_list = list(BRICS_ENV.keys())

    possible_index_list: List[Tuple[int, str]] = []
    for brics_label in brics_label_list :
        substructure = BRICS_substructure[brics_label]
        for idxs_list in core.GetSubstructMatches(substructure) :
            atom_idx = idxs_list[0]
            possible_index_list.append((atom_idx, brics_label))
    return possible_index_list 

def get_possible_brics_labels(core: Union[SMILES, Mol],
                              atom_idx: Optional[int] = None) -> List[str] :
    def fast_brics_search(atom: Atom) :
        atomicnum = atom.GetAtomicNum()
        aromatic = atom.GetIsAromatic()
        if atomicnum == 6 :
            if aromatic :
                return ['14', '16']
            else :
                return ['1', '4', '6', '7', '8', '13', '15']
        elif atomicnum == 7 :
            if aromatic :
                return ['9']
            else :
                return ['5', '10']
        elif atomicnum == 8 : 
            return ['3']
        elif atomicnum == 16 :
            return ['11', '12']
        else :
            return []

    core: Mol = convert_to_rdmol(core)
    
    if atom_idx is not None :
        brics_label_list = fast_brics_search(core.GetAtomWithIdx(atom_idx))
    else :
        brics_label_list = list(BRICS_ENV.keys())

    possible_brics_label_list: List[str] = []
    for brics_label in brics_label_list :
        substructure = BRICS_substructure[brics_label]
        for idxs_list in core.GetSubstructMatches(substructure) :
            __atom_idx = idxs_list[0]
            if atom_idx is None or atom_idx == __atom_idx:
                possible_brics_label_list.append(brics_label)
                break

    return possible_brics_label_list 
