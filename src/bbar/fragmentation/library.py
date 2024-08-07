import torch
from torch import FloatTensor

from rdkit import Chem

import os
import logging

from typing import Union, List, Optional, Tuple
from rdkit.Chem import Mol
from bbar.utils.typing import SMILES

from bbar.utils.common import convert_to_SMILES 
from .fragmentation import Fragmentation

class BlockLibrary() : 
    fragmentation = Fragmentation()
    def __init__(
        self,
        library_path: Optional[str] = None,
        smiles_list: Optional[List[SMILES]] = None,
        frequency_list: Optional[FloatTensor] = None,
        use_frequency: bool = True,
        save_rdmol: bool = False,
    ) :
        
        if library_path is not None :
            smiles_list, frequency_list = self.load_library_file(library_path, use_frequency)

        assert smiles_list is not None
        if not use_frequency :
            frequency_list = None

        self._smiles_list = smiles_list

        if frequency_list is not None:
            self._frequency_distribution = frequency_list 
        else :
            if use_frequency :
                logging.warning(f'No Frequency Information in library file ({library_path})')
            self._frequency_distribution = torch.full((len(smiles_list),), 1/len(smiles_list))

        self._smiles_to_index = {smiles: index \
                                    for index, smiles in enumerate(smiles_list)}

        # Set RDMol List
        if save_rdmol :
            self._rdmol_list = [Chem.MolFromSmiles(s) for s in smiles_list]
        else :
            self._rdmol_list = None

    def __len__(self) -> int:
        return len(self._smiles_list)

    def __getitem__(self, index: int) -> SMILES :
        return self._smiles_list[index]

    def get_smiles(self, index: int) -> SMILES :
        return self._smiles_list[index]

    def get_rdmol(self, index: int) -> Mol :
        if self._rdmol_list is not None :
            return self._rdmol_list[index]
        else :
            return Chem.MolFromSmiles(self.get_smiles(index))

    def get_index(self, mol: Union[SMILES, Mol]) -> int :
        smiles = convert_to_SMILES(mol)
        return self._smiles_to_index[smiles]
        
    @property
    def smiles_list(self) :
        return self._smiles_list

    @property
    def rdmol_list(self) -> List[Mol]:
        if self._rdmol_list is not None :
            return self._rdmol_list
        else :
            return [Chem.MolFromSmiles(smiles) for smiles in self._smiles_list]

    @property
    def frequency_distribution(self) -> FloatTensor:
        return self._frequency_distribution

    def load_library_file(self, library_path: str, use_frequency: True) \
                                        -> Tuple[List[SMILES], FloatTensor] :
        extension = os.path.splitext(library_path)[1]
        assert extension in ['.smi', '.csv'], \
            "Extension of library file should be '.smi' or '.csv'\n" + \
            f"Current file name: {library_path}"

        with open(library_path) as f :
            frequency_list = None
            if extension == '.smi' :
                smiles_list = [l.strip() for l in f.readlines()]
            else :
                header = f.readline().split(',')
                if len(header) == 1 :
                    smiles_list = [l.strip() for l in f.readlines()]
                else :
                    lines = [l.strip().split(',') for l in f.readlines()]
                    smiles_list = [smiles for smiles, _ in lines]
                    if use_frequency :
                        frequency_list = torch.FloatTensor([float(frequency) for _, frequency in lines])

        return smiles_list, frequency_list

    @classmethod
    def create_library_file(
        cls,
        library_path: str,
        mol_list: List[Union[Mol, SMILES]],
        save_frequencey: bool = True,
        cpus: int = 1,
    ) :
        from collections import Counter
        import parmap

        """
        assertion: extension of library path should be 'csv' or 'smi'
        """
        extension = os.path.splitext(library_path)[1]
        assert extension in ['.smi', '.csv'], \
            "Extension of library file should be '.smi' or '.csv'\n" + \
            f"Current file name: {library_path}"

        res = parmap.map(cls.fragmentation.decompose, mol_list, pm_processes = cpus, pm_chunksize=1000, pm_pbar=True)
        block_list: List[SMILES] = []
        for blocks in res :
            block_list += blocks
        block_freq_list = sorted(Counter(block_list).items(), key=lambda item: item[1], reverse=True)

        if save_frequencey is not None :
            assert extension == '.csv'
            with open(library_path, 'w') as w:
                w.write('SMILES,frequency\n')
                for block, freq in block_freq_list :
                    block = convert_to_SMILES(block)
                    w.write(f'{block},{freq}\n')
        else :
            with open(library_path, 'w') as w :
                if extension == '.csv' :
                    w.write('SMILES\n')
                for block, _ in block_freq_list :
                    block = convert_to_SMILES(block)
                    w.write(block + '\n')
