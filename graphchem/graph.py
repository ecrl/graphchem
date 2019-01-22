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

# Number of valence electons available for each atom
NUM_VALENCE = {'C': 4, 'O': 2}

# Correlation between number of bonds and bond angle
BOND_ANGLES = {1: 360, 2: 180, 3: 120, 4: 109.5}

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
        self.state = None
        self._id = id
        self._char = char
        self._bonds = []
        self._branch_lvl = branch_lvl
        self._link = None

    def initialize(self):
        '''Creates vector representations for the atom, its bonds, and its
        bond angle

        TODO: add more indices for atom distinctions
        '''

        bonds = [b[1] for b in self.connections]
        self._bonds = [
            sum([b[0] for b in bonds]),
            sum([b[1] for b in bonds]),
            sum([b[2] for b in bonds]),
            sum([b[3] for b in bonds]),
            sum([b[4] for b in bonds])
        ]

        bond_angle = BOND_ANGLES[(
            NUM_VALENCE[self._char] -
            (
                self._bonds[0] +
                2 * self._bonds[1] +
                3 * self._bonds[2]
            ) + sum(self._bonds)
        )] / 360

        self.state = []
        self.state.extend([a for a in ATOMS[self._char]])
        self.state.append(bond_angle)
        self.state.extend([b for b in self._bonds])


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
        self.reset_graph()

    def __len__(self):

        return len(self.atoms)

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

    @property
    def feed_len(self):
        '''Returns the length of each atom's feed, i.e. sum of neighbor states
        appended to current state
        '''

        return 2 * len(self.atoms[0].state)

    @property
    def state_len(self):
        '''Returns the length of each atom's state'''

        return len(self.atoms[0].state)

    @property
    def feeds(self):
        '''Returns all atoms' previous feeds (each timestep is generated from
        self.propagate using supplied transition function)
        '''

        return self.__prev_feeds

    @property
    def states(self):
        '''Returns all atoms' states'''

        return [a.state for a in self.atoms]

    def reset_graph(self):

        self.__prev_feeds = [[] for _ in range(len(self.atoms))]
        for a in self.atoms:
            a.initialize()

    def propagate(self, transition_fn):
        '''Updates the states of all atoms using supplied function

        Args:
            transition_fn (callable): function to perform operation; this
                function must return a single value, int or float
        '''

        if not callable(transition_fn):
            raise ReferenceError('Supplied `transition_fn` not callable')

        feed = []
        for idx, a in enumerate(self.atoms):
            a_feed = [i for i in a.state]
            a_feed.extend(self._stack(
                [self.atoms[c[0]].state for c in a.connections]
            ))
            feed.append(a_feed)
            self.__prev_feeds[idx].append(a_feed)
        new_states = transition_fn(feed)
        for idx, a in enumerate(self.atoms):
            if len(new_states[idx]) != len(a.state):
                raise RuntimeError(
                    'Transition function did not return a state equal to the'
                    ' length of the current state: {}, {}'.format(
                        len(new_states[idx]),
                        len(a.state)
                    )
                )
            a.state = new_states[idx]

    @staticmethod
    def _stack(it):
        '''Stacks each sub-iterable's values by index (similar to tf.reduce_sum)

        Args:
            it (iterable): iterable of iterables, where each sub-iterable is
                of equal length and is populated with ints or floats
        Returns:
            list: single-dimension list of stacked items
        '''

        stack = []
        for i in range(len(it[0])):
            stack.append(sum([s[i] for s in it]))
        return stack

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