#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# atom.py
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#

# custom imports
from graphchem.chem.atom_features import ATOM_VECTORS, BOND_VECTORS


class Atom:

    def __init__(self, id, symbol, branch_level):
        '''Atom object: houses atom ID, symbol, connections to neighboring
        atoms, and atom state

        Args:
            id (int): ID of the atom
            symbol (str): atom symbol
            branch_level (int): used by Molecule._construct to determine
                branching atoms
        '''

        self.id = id
        self.symbol = symbol
        self._branch_level = branch_level
        self._connections = []
        self._link = None
        self._state = None

    def add_connection(self, atom, bond_symbol):
        '''Adds an atom to the list of this atom's connections

        Args:
            atom (Atom): neighboring atom
            bond_symbol (string): SMILES bond symbol for the connection
        '''

        self._connections.append(
            (atom, BOND_VECTORS[bond_symbol], bond_symbol)
        )

    @property
    def state(self):
        '''Atom state: returns a list with the atom type (binary vector for
        identifying different atoms, e.g. carbon/nitrogen), sum of all bond
        types (sum of binary vectors identifying different bonds, e.g. single,
        double), and whether the atom exists in a ring or not ([1, 0] or [0, 1]
        respectively). If the state was updated externally, returns the
        updated state.

        Returns:
            list: atom state
        '''

        if self._state is None:
           self._state = self._gen_state()
           neighbor_states = [c[0]._gen_state() for c in self._connections]
           self._state.extend([sum(s) for s in zip(*neighbor_states)])
        return self._state

    def _gen_state(self):

        state = [a for a in ATOM_VECTORS[self.symbol]]
        bonds = [c[1] for c in self._connections]
        for i in range(len(bonds[0])):
            sum = 0
            for b in bonds:
                sum += b[i]
            state.append(sum)
        if self.in_ring:
            state.extend([1, 0])
        else:
            state.extend([0, 1])
        return state

    @property
    def connectivity(self):

        return [c[0].id for c in self._connections]

    @property
    def in_ring(self):
        '''Recursively determines whether the atom exists in a ring

        Returns:
            bool: True if in a ring, False if not in a ring
        '''

        visited_atoms = []

        def recursive_atom_search(atom, base_id):
            visited_atoms.append(atom.id)
            prev_atom = base_id
            for con in atom._connections:
                if con[0].id == self.id and prev_atom != self.id:
                    return True
                elif con[0].id in visited_atoms:
                    continue
                else:
                    found_self = recursive_atom_search(con[0], atom.id)
                    if found_self:
                        return True
            return False

        return recursive_atom_search(self, self.id)
