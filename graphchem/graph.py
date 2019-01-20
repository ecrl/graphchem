#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph.py
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#

# Stdlib imports
from re import compile
from statistics import stdev

from numpy import asarray

# Vector representation for each atom
ATOMS = {'C': [1, 0], 'O': [0, 1]}

# Vector representation for each bond type
BONDS = {
    '-': [1, 0, 0, 0, 0],
    '=': [0, 1, 0, 0, 0],
    '#': [0, 0, 1, 0, 0],
    '*': [0, 0, 0, 1, 0],
    '.': [0, 0, 0, 0, 1]
}

# Names of each bond type
BOND_NAMES = {
    '-': 'Single',
    '=': 'Double',
    '#': 'Triple',
    '*': 'Aromatic',
    '.': 'Disconnected'
}

# RegEx for finding link characters
LINK = compile(r'\d')


class _Atom:

    def __init__(self, id, char, branch_lvl):
        '''Atom object: represents an atom and its connections/bonds

        Args:
            id (int): identifier for the atom
            char (str): symbol for the atom
            branch_lvl (int): used by Graph._connect to determine branch
                connections
        '''

        self.connections = []
        self.state = 0
        self._id = id
        self._char = char
        self._bonds = []
        self._branch_lvl = branch_lvl
        self._link = None

    def initialize(self):
        '''Creates vector representations for the atom's bonds
        '''

        bonds = [b[1] for b in self.connections]
        self._bonds = [
            sum([b[0] for b in bonds]),
            sum([b[1] for b in bonds]),
            sum([b[2] for b in bonds]),
            sum([b[3] for b in bonds]),
            sum([b[4] for b in bonds])
        ]

    def pack(self):
        '''Returns:
            tuple: (carbon, oxygen, number of single bonds, number of double
                bonds, number of triple bonds, number of aromatic bonds,
                number of disconnected bonds)
        '''

        packed = [a for a in ATOMS[self._char]]
        packed.extend([b for b in self._bonds])
        return tuple(packed)


class Graph:

    def __init__(self, smiles):
        '''Graph object to represent atom connections for a given molecule

        Args:
            smiles (str): SMILES representation for a molecule; currently
                supports carbon and oxygen atoms, and single, double, triple,
                aromatic and disconnected bond types
        '''

        for idx, char in enumerate(smiles):
            if char not in list(ATOMS.keys()) \
             and char not in list(BONDS.keys()) \
             and LINK.match(char) is None \
             and char not in ['(', ')']:
                raise ValueError('Invalid SMILES character: {}'.format(
                    char
                ))
            if LINK.match(char) is not None and idx == 0:
                raise ValueError('Invalid link placement for {}'.format(
                    char
                ))
        self.atoms = self._construct(smiles)
        self.smiles = smiles
        for atom in self.atoms:
            atom.initialize()
        self.__prev_feeds = []

    def __len__(self):

        return len(self.atoms)

    @staticmethod
    def _construct(smiles):
        '''Creates _Atom objects for supplied molecule

        Args:
            smiles (str): molecule's SMILES string

        Returns:
            list: each element is an _Atom object
        '''

        atoms = []
        branch_lvl = 0
        bond_type = BONDS['-']
        bond_name = BOND_NAMES['-']
        _offset = 0
        _new_branch = False
        for idx, char in enumerate(smiles):
            if char == '(':
                branch_lvl += 1
                _offset += 1
                _new_branch = True
            elif char == ')':
                branch_lvl -= 1
                _offset += 1
                _new_branch = False
            elif char in list(BONDS.keys()):
                bond_type = BONDS[char]
                _offset += 1
                bond_name = BOND_NAMES[char]
            elif LINK.match(char) is not None:
                _offset += 1
                atoms[-1]._link = int(char)
                for li, atom in enumerate(atoms[0: -1]):
                    if atom._link == int(char):
                        atom.connections.append(
                            (idx - _offset, bond_type, bond_name)
                        )
                        atoms[-1].connections.append(
                            (li, bond_type, bond_name)
                        )
                        break
            else:
                new_atom = _Atom(idx - _offset, char, branch_lvl)
                if idx > 0:
                    for i in range(1, len(atoms) + 1):
                        if atoms[-1 * i]._branch_lvl == branch_lvl:
                            if _new_branch:
                                continue
                            else:
                                atoms[-1 * i].connections.append(
                                    (idx - _offset, bond_type, bond_name)
                                )
                                new_atom.connections.append(
                                    (len(atoms) - i, bond_type, bond_name)
                                )
                                break
                        elif atoms[-1 * i]._branch_lvl < branch_lvl:
                            atoms[-1 * i].connections.append(
                                (idx - _offset, bond_type, bond_name)
                            )
                            new_atom.connections.append(
                                (len(atoms) - i, bond_type, bond_name)
                            )
                            _new_branch = False
                            break
                atoms.append(new_atom)
                bond_type = BONDS['-']
                bond_name = BOND_NAMES['-']
        return atoms

    def __repr__(self):
        '''Returns graph representation, with each atom's ID, atom symbol and
        connections
        '''

        r = 'ID\tAtom\tConnections\n'
        for atom in self.atoms:
            r += '{}\t{}\t{}\n'.format(
                atom._id,
                atom._char,
                [(c[0], c[2]) for c in atom.connections]
            )
        return r

    def pack(self):
        '''Returns:
            list: list of tuples, where each tuple is an atom's vector
                representation in the form:
            (carbon, oxygen, number of single bonds, number of double bonds,
            number of triple bonds, number of aromatic bonds, number of
            disconnected bonds)
        '''

        return [a.pack() for a in self.atoms]

    def reset_states(self):
        '''Reset atom states (atom.state == 0) and all time-series info'''

        for atom in self.atoms:
            atom.state = 0
        self.__prev_feeds = []

    @property
    def states(self):
        '''Returns:
            tuple: states of all atoms
        '''

        return tuple([a.state for a in self.atoms])

    @property
    def current_repr(self):
        '''Returns:
            list: current transition function feed lists for all atoms
        '''

        return [self._get_feed(a) for a in self.atoms]

    @property
    def prev_repr(self):
        '''Returns:
            list: previous transition function feed lists for all previous
                propagations of each atom
        '''

        return self.__prev_feeds

    @property
    def model_repr(self):

        model_repr = []
        for a in range(len(self.__prev_feeds[0])):
            atom = []
            for propagation in self.__prev_feeds:
                atom.append(asarray(propagation[a]))
            model_repr.append(atom)
        return asarray(model_repr)

    def propagate(self, transition_fn):
        '''Updates the states of all atoms using supplied function

        Args:
            transition_fn (callable): function to perform operation; this
                function must return a single value, int or float
        '''

        if not callable(transition_fn):
            raise ReferenceError('Supplied `transition_fn` not callable')

        new_states = []
        prev_feed = []

        for atom in self.atoms:
            f = self._get_feed(atom)
            prev_feed.append(f)
            new_states.append(transition_fn([f]))

        for idx, atom in enumerate(self.atoms):
            atom.state = new_states[idx]
        self.__prev_feeds.append(prev_feed)

    def _get_feed(self, atom):
        '''Obtains transition function feed lists for supplied atom

        Args:
            atom (_Atom): atom object
        '''

        neighbor_atoms = [self.atoms[a[0]] for a in atom.connections]
        neighbor_states = [a.state for a in neighbor_atoms]
        neighbor_repr = [a.pack() for a in neighbor_atoms]
        states_stack = self._stack(neighbor_states)
        repr_stack = self._vert_stack(neighbor_repr)
        return (states_stack,) + repr_stack

    @staticmethod
    def _stack(it):
        '''Computes a "stack" from items in supplied iterable

        NOTE: this can probably be improved

        Args:
            it (iterable): iterable values, int or float
        '''

        return sum(it)

    def _vert_stack(self, it):
        '''Creates "stacks" for each iterable item's values, stacking all the
        iterable's item's values by index

        Args:
            it (iterable): iterable of iterables, where each sub-iterable is
                of equal length and is populated with ints or floats
        '''

        stack = []
        for i in range(len(it[0])):
            stack.append(self._stack([s[i] for s in it]))
        return tuple(stack)
