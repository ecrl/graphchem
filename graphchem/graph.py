#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph.py
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#

# Stdlib imports
from re import compile

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


class _Node:

    def __init__(self, id, atom, branch_lvl):
        '''Node object: represents an atom and its connections/bonds

        Args:
            id (int): identifier for the atom
            atom (str): symbol for the atom
            branch_lvl (int): used by Graph._connect to determine branch
                connections
        '''

        self.connections = []
        self._id = id
        self._atom = atom
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

        packed = [a for a in ATOMS[self._atom]]
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
        self.smiles = smiles
        self.nodes = self._connect(smiles)
        self.__initialize()

    def __repr__(self):
        '''Returns graph representation, with each atom's ID, atom symbol and
        connections
        '''

        r = 'ID\tAtom\tConnections\n'
        for node in self.nodes:
            r += '{}\t{}\t{}\n'.format(
                node._id,
                node._atom,
                [(c[0], c[2]) for c in node.connections]
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

        return [n.pack() for n in self.nodes]

    @staticmethod
    def _connect(smiles):
        '''Static method: creates graph nodes from each SMILES character

        Returns:
            list: list of _Node objects, where each node object represents an
                atom and its bonds/connections
        '''

        nodes = []
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
                nodes[-1]._link = int(char)
                for li, node in enumerate(nodes[0: -1]):
                    if node._link == int(char):
                        node.connections.append(
                            (idx - _offset, bond_type, bond_name)
                        )
                        nodes[-1].connections.append(
                            (li, bond_type, bond_name)
                        )
                        break
            else:
                new_node = _Node(idx - _offset, char, branch_lvl)
                if idx > 0:
                    for i in range(1, len(nodes) + 1):
                        if nodes[-1 * i]._branch_lvl == branch_lvl:
                            if _new_branch:
                                continue
                            else:
                                nodes[-1 * i].connections.append(
                                    (idx - _offset, bond_type, bond_name)
                                )
                                new_node.connections.append(
                                    (len(nodes) - i, bond_type, bond_name)
                                )
                                break
                        elif nodes[-1 * i]._branch_lvl < branch_lvl:
                            nodes[-1 * i].connections.append(
                                (idx - _offset, bond_type, bond_name)
                            )
                            new_node.connections.append(
                                (len(nodes) - i, bond_type, bond_name)
                            )
                            _new_branch = False
                            break
                nodes.append(new_node)
                bond_type = BONDS['-']
                bond_name = BOND_NAMES['-']
        return nodes

    def __initialize(self):
        '''Private method: initializes each _Node/atom in self.nodes, creating
        vector representations for each atom's bonds
        '''

        for node in self.nodes:
            node.initialize()
