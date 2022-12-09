from rdkit import Chem

import random
from itertools import combinations

from typing import Optional, Union, List, Tuple, Set
from rdkit.Chem import Mol, Atom, Bond, BondType
from bbar.utils.typing import SMILES

from . import utils
from bbar.utils.common import convert_to_rdmol

__all__ = ['Fragmentation', 'fragmentation'] 

"""
Scaffold: Substructure of molecule
Fragment: Minimum unit of fragmentation. It contains dummy atom.
dummy atom includes information for merging scaffold and fragment. (bond-type, label(Optional))
ex)
 - *c1ccccc1 (*-benzene)
 - *=Cc1cccc1 : dummy atom is connected with double bond.

Notation
atom_index: atom index of original molecule
new_atom_index: atom index of new(sub) molecule
atom_indices: [atom_index]  ( equal to {new_atom_index: atom_index} )
bond_indices: [bond_index]  ( rdkit.Chem.Bond -> GetBondIdx() )
atomMap: {atom_index: new_atom_index}
"""

class Unit() :
    def __init__(self, graph, atom_indices: Tuple[int]) :
        self.graph = graph

        atom_indices_set = set(atom_indices)
        bond_indices = []
        for bond in graph.rdmol.GetBonds() :
            atom_idx1, atom_idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if atom_idx1 in atom_indices_set and atom_idx2 in atom_indices_set :
                bond_indices.append(bond.GetIdx())

        self.atom_indices = atom_indices
        self.bond_indices = tuple(bond_indices)

        self.neighbors = []
        self.connections = []
        
    def add_connection(self, neighbor_unit, connection) :
        self.neighbors.append(neighbor_unit)
        self.connections.append(connection)
        
    def to_rdmol(self) -> Mol :
        return self.graph.get_submol([self])

    def to_fragment(self, connection) -> Mol :
        assert connection in self.connections
        atomMap = {}
        submol = self.graph.get_submol([self], atomMap = atomMap)
        if self == connection.units[0] :
            atom_index = atomMap[connection.atom_indices[0]]
        else :
            atom_index = atomMap[connection.atom_indices[1]]
        bondtype = connection.bondtype

        rwmol = Chem.RWMol(submol)
        utils.add_dummy_atom(rwmol, atom_index, bondtype)
        fragment = rwmol.GetMol()
        fragment = Chem.MolFromSmiles(Chem.MolToSmiles(fragment))
        return fragment

class Connection() :
    def __init__(self, unit1: Unit, unit2: Unit, atom_index1: int, atom_index2: int, bond_index: int, bondtype: BondType) :
        # bond_index: rdkit.Chem.Bond->GetIdx()
        self.units = (unit1, unit2)
        self.atom_indices = (atom_index1, atom_index2)
        self.bond_index = bond_index
        self._bondtype = int(bondtype)
        unit1.add_connection(unit2, self)
        unit2.add_connection(unit1, self)
    
    @property
    def bondtype(self) :
        return BondType.values[self._bondtype]

class FragmentedGraph() :
    def fragmentation(self, mol: Mol) -> Tuple[Tuple[Unit], Tuple[Connection]] :
        raise NotImplementedError
        # return units, connections

    def __init__(self, mol: Union[SMILES, Mol]) :
        rdmol = convert_to_rdmol(mol, isomericSmiles=False)
        self.rdmol = rdmol

        units, connections = self.fragmentation(rdmol)
        self.units = units
        self.num_units = len(units)

        self.connections = connections
        self.connection_dict = {}
        for connection in connections :
            unit1, unit2 = connection.units
            self.connection_dict[(unit1, unit2)] = connection
            self.connection_dict[(unit2, unit1)] = connection

    def __len__(self) :
        return self.num_units

    def get_submol(self, unit_list: List[Unit], atomMap: dict = {}) -> Mol :
        atom_indices = []
        bond_indices = []
        for unit in unit_list :
            atom_indices+=unit.atom_indices
            bond_indices+=unit.bond_indices
        for unit1, unit2 in combinations(unit_list, 2) :
            connection = self.connection_dict.get((unit1, unit2), None)
            if connection is not None :
                bond_indices.append(connection.bond_index)

        atomMap.update({atom_index: new_atom_index for new_atom_index, atom_index in enumerate(atom_indices)})

        rwmol = Chem.RWMol()
        src_atom_list = [self.rdmol.GetAtomWithIdx(atom_index) for atom_index in atom_indices]
        src_bond_list = [self.rdmol.GetBondWithIdx(bond_index) for bond_index in bond_indices]
        for src_atom in src_atom_list :
            rwmol.AddAtom(src_atom)

        for src_bond in src_bond_list :
            src_atom_index1, src_atom_index2 = src_bond.GetBeginAtomIdx(), src_bond.GetEndAtomIdx()
            dst_atom_index1, dst_atom_index2 = atomMap[src_atom_index1], atomMap[src_atom_index2]
            bondtype = src_bond.GetBondType()
            rwmol.AddBond(dst_atom_index1, dst_atom_index2, bondtype)
        
        # Update Atom Feature
        for src_atom, dst_atom in zip(src_atom_list, rwmol.GetAtoms()) :
            if dst_atom.GetAtomicNum() == 7 :
                degree_diff = src_atom.GetDegree() - dst_atom.GetDegree()
                if degree_diff > 0 :
                    dst_atom.SetNumExplicitHs(dst_atom.GetNumExplicitHs() + degree_diff)

        submol = rwmol.GetMol()
        Chem.SanitizeMol(submol)

        return submol

    def get_datapoint(self, traj = None) -> Tuple[Mol, Mol, Tuple[int, int]] :
        """
        trajectory (sub-trajectory)
            - [C,D] => scaffold: C, fragment: (*-D)
            - [B,A,C] => scaffold: A-B, fragment: (*-C)
            - [A,B,C,D,None] => scaffold: A-B-C-D, fragment: None (Termination)
            - ...

        return:
            scaffold: Mol
            fragment: Mol which contains dummy atom
            connection: (int, int) => (scaffold atom index, fragment atom index)
        """
        if traj is None :
            traj = self.get_random_subtrajectory(min_length = 2)
        scaffold_units, fragment_unit = traj[:-1], traj[-1]

        if fragment_unit is None :
            scaffold = Chem.Mol(self.rdmol)
            return scaffold, None, (None, None)
        else :
            # find Connection between scaffold_units and fragment_unit
            neighbor_units = set(fragment_unit.neighbors).intersection(set(scaffold_units))
            assert len(neighbor_units) == 1
            neighbor_unit = neighbor_units.pop()
            connection = self.connection_dict[(fragment_unit, neighbor_unit)]

            # get scaffold and fragment
            atomMap = {}
            scaffold = self.get_submol(scaffold_units, atomMap=atomMap)
            fragment = fragment_unit.to_fragment(connection)

            # get atom index pair to represent bond between scaffold and fragment
            if fragment_unit is connection.units[0] :
                scaffold_atom_index = atomMap[connection.atom_indices[1]]
            else :
                scaffold_atom_index = atomMap[connection.atom_indices[0]]
            fragment_atom_index = fragment.GetNumAtoms() - 1   # atom index of dummy atom
            
            return scaffold, fragment, (scaffold_atom_index, fragment_atom_index)

    def get_random_trajectory(self) -> List[Unit] :
        """
        trajectory: 
            choose random unit A
            choose random unit B adjacent to A
            choose random unit C adjacent to [A,B]
            ...
            At the end, add termination sign `None`

        ex) A-B-C-D
            - [C,D,B,A,None]
            - [B,A,C,D,None]
            - [A,B,C,D,None]
            - ...
        """
        return self.get_random_subtrajectory(min_length = self.num_units + 1)
        
    def get_random_subtrajectory(
        self,
        min_length = 1,
        max_length = None,
    ) -> List[Unit]:
        """
        sub-trajectory
        ex) A-B-C-D
            - [C,D]
            - [B,A,C]
            - [A,B,C,D,None]    # None: termination sign
            - ...
        trajectory length is random value btw min_length and N+1     # N: num of unit
        """
        assert max_length is None or max_length >= min_length
        if max_length is None :
            max_length = self.num_units + 1 
        traj_length = random.randrange(min_length, max_length+1)

        if traj_length == self.num_units + 1:
            traj = list(self.units) + [None]
        else :
            start_unit = random.choice(self.units)
            traj = [start_unit]
            neighbors = set(start_unit.neighbors)
            for _ in range(traj_length - 1) :   # minus one because already one unit in trajectory (traj = [start_unit])
                unit = random.choice(tuple(neighbors))
                traj.append(unit)
                neighbors.update(unit.neighbors)
                neighbors = neighbors.difference(traj)
        return traj 

def fragmentation(mol: Union[SMILES, Mol]) -> FragmentedGraph:
    return FragmentedGraph(mol)

class Fragmentation() :
    fragmentation = staticmethod(fragmentation)
    def __call__(self, mol: Union[SMILES, Mol]) -> FragmentedGraph:
        return self.fragmentation(mol)
    
    @staticmethod
    def merge(scaffold: Mol, fragment: Mol, scaffold_atom_index, fragment_atom_index) -> Mol:
        return utils.merge(scaffold, fragment, scaffold_atom_index, fragment_atom_index)

    @classmethod
    def decompose(cls, mol: Union[SMILES, Mol]) -> List[SMILES]:
        rdmol = convert_to_rdmol(mol)
        fragmented_mol = cls.fragmentation(rdmol)
        if len(fragmented_mol) == 1 :
            fragments = []
        else :
            fragments = [
                Chem.MolToSmiles(unit.to_fragment(connection)) \
                                    for unit in fragmented_mol.units \
                                    for connection in unit.connections
            ]
        return fragments

