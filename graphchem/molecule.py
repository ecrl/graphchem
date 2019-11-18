#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# molecule.py
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#

# custom imports
from graphchem.atom import Atom
from graphchem.atom_features import ATOM_VECTORS, BOND_VECTORS

# stdlib. imports
from re import compile, IGNORECASE
from shutil import which
from os import devnull, remove
from subprocess import call


class Molecule:

    def __init__(self, smiles):
        '''Create a graph representation of a molecule from its SMILES string

        Args:
            smiles (str): SMILES string for the molecule
        '''

        self.smiles = smiles
        self._atoms = self._construct(smiles)

    def __repr__(self):
        '''Returns graph representation, with each atom's ID, atom symbol and
        connections
        '''

        r = 'ID\tAtom\tConnections\n'
        for atom in self._atoms:
            r += '{}\t{}\t{}\n'.format(
                atom.id,
                atom.symbol,
                [(c[0].id, c[2]) for c in atom._connections]
            )
        return r

    def to_mdl(self, filename):
        '''Saves the molecule to an MDL file (requires Open Babel to be installed)

        Args:
            filename (str): path to save MDL file, requires `.mdl` extension
        '''

        if which('obabel') is None:
            raise ReferenceError('Open Babel installation not found')

        is_mdl = compile(r'.*\.mdl$', IGNORECASE)
        if is_mdl.match(filename) is None:
            raise ValueError('Output file must have an MDL extension: {}'
                             .format(filename))

        with open('_smi_temp', 'w') as temp_smi_file:
            temp_smi_file.write(self.smiles)
        temp_smi_file.close()

        dn = open(devnull, 'w')
        call(['obabel', '-i', 'smi', '_smi_temp',
              '-o', 'mdl', '-O', filename, '--gen3D'],
             stdout=dn, stderr=dn, timeout=10)

        remove('_smi_temp')

    @staticmethod
    def _construct(smiles):
        '''Constructs the molecule structure from supplied SMILES string

        Args:
            smiles (str): SMILES string for a molecule

        Returns:
            list: list of `Atom` objects
        '''

        # numbers indicate a link/ring
        link_re = compile(r'\d')
        atoms = []
        atom_count = 0

        # initialize branch level to 0
        branch_level = 0
        new_branch = False

        # initialise bond type to single-bond
        bond_type = '-'

        # iterate over all characters in SMILES string
        for char in smiles:

            # open parentheses indicate a new branch
            if char == '(':
                branch_level += 1
                new_branch = True

            # close parentheses indicate the end of a branch
            elif char == ')':
                branch_level -= 1
                new_branch = False

            # if bond character, adjust bond type accordingly
            elif char in list(BOND_VECTORS.keys()):
                bond_type = char

            # if link/ring (integer), connect other atoms in ring
            elif link_re.match(char) is not None:
                link = int(char)
                atoms[-1]._link = link
                for atom in atoms[:-1]:
                    if atom._link == link:
                        atom.add_connection(atoms[-1], bond_type)
                        atoms[-1].add_connection(atom, bond_type)

            # character is an atom symbol
            elif char in list(ATOM_VECTORS.keys()):

                # create a new atom
                new_atom = Atom(atom_count, char, branch_level)
                atom_count += 1

                # add connection to previous atom(s), checking branch levels
                for atom in reversed(atoms):
                    if atom._branch_level == branch_level:
                        if new_branch:
                            continue
                        else:
                            atom.add_connection(new_atom, bond_type)
                            new_atom.add_connection(atom, bond_type)
                            break
                    elif atom._branch_level < branch_level:
                        atom.add_connection(new_atom, bond_type)
                        new_atom.add_connection(atom, bond_type)
                        new_branch = False
                        break
                atoms.append(new_atom)
                bond_type = '-'

            # unknown character
            else:
                raise ValueError('Unknown SMILES character: {}'.format(char))
        return atoms
