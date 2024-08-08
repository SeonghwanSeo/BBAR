from rdkit import Chem
from rdkit.Chem import BRICS

from typing import Union, List, Tuple
from rdkit.Chem import Mol, BondType

from . import utils
from .fragmentation import Unit, Connection, FragmentedGraph, Fragmentation
from .library import BlockLibrary

__all__ = ["BRICS_FragmentedGraph", "BRICS_Fragmentation", "brics_fragmentation", "BRICS_BlockLibrary"]


class BRICS_Unit(Unit):
    def to_fragment(self, connection):
        assert connection in self.connections
        atomMap = {}
        submol = self.graph.get_submol([self], atomMap=atomMap)
        if self == connection.units[0]:
            atom_index = atomMap[connection.atom_indices[0]]
            brics_label = connection.brics_labels[0]
        else:
            atom_index = atomMap[connection.atom_indices[1]]
            brics_label = connection.brics_labels[1]
        bondtype = connection.bondtype

        rwmol = Chem.RWMol(submol)
        utils.add_dummy_atom(rwmol, atom_index, bondtype, brics_label)
        fragment = rwmol.GetMol()
        Chem.SanitizeMol(fragment)
        return fragment


class BRICS_Connection(Connection):
    def __init__(
        self,
        unit1: Unit,
        unit2: Unit,
        atom_index1: int,
        atom_index2: int,
        brics_label1: Union[str, int],
        brics_label2: Union[str, int],
        bond_index: int,
        bondtype: BondType,
    ):
        super().__init__(unit1, unit2, atom_index1, atom_index2, bond_index, bondtype)
        self.brics_labels = (int(brics_label1), int(brics_label2))


class BRICS_FragmentedGraph(FragmentedGraph):
    def fragmentation(self, mol: Mol) -> Tuple[List[Unit], List[Connection]]:
        brics_bonds = list(BRICS.FindBRICSBonds(mol))

        rwmol = Chem.RWMol(mol)
        for (atom_idx1, atom_idx2), _ in brics_bonds:
            utils.remove_bond(rwmol, atom_idx1, atom_idx2)
        broken_mol = rwmol.GetMol()

        atomMap = Chem.GetMolFrags(broken_mol)
        units = tuple(BRICS_Unit(self, atom_indices) for atom_indices in atomMap)

        unit_map = {}
        for unit in units:
            for idx in unit.atom_indices:
                unit_map[idx] = unit

        connections = []
        for brics_bond in brics_bonds:
            (atom_index1, atom_index2), (brics_label1, brics_label2) = brics_bond
            bond = mol.GetBondBetweenAtoms(atom_index1, atom_index2)
            assert bond is not None
            bond_index, bondtype = bond.GetIdx(), bond.GetBondType()
            unit1 = unit_map[atom_index1]
            unit2 = unit_map[atom_index2]
            connection = BRICS_Connection(
                unit1, unit2, atom_index1, atom_index2, brics_label1, brics_label2, bond_index, bondtype
            )
            connections.append(connection)
        connections = tuple(connections)

        return units, connections


def brics_fragmentation(mol) -> BRICS_FragmentedGraph:
    return BRICS_FragmentedGraph(mol)


class BRICS_Fragmentation(Fragmentation):
    fragmentation = staticmethod(brics_fragmentation)


class BRICS_BlockLibrary(BlockLibrary):
    fragmentation = BRICS_Fragmentation()

    @property
    def brics_label_list(self):
        def get_brics_label(rdmol: Mol):
            return str(rdmol.GetAtomWithIdx(0).GetIsotope())

        brics_labels: List[str] = [get_brics_label(rdmol) for rdmol in self.rdmol_list]
        return brics_labels
