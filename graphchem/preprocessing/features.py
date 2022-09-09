from typing import List, Tuple, Union

import rdkit
from rdkit.Chem import MolFromSmiles
import numpy as np
import torch
from torchnlp.encoders.text import DelimiterEncoder


def get_ring_size(obj: Union['rdkit.Chem.Atom', 'rdkit.Chem.Bond'],
                  max_size: int = 12):
    """ determine whether rdkit.Chem.Atom or rdkit.Chem.Bond is in a ring, and
    of which size

    Args:
        obj (Union[rdkit.Chem.Atom, rdkit.Chem.Bond]): atom or bond
        max_size (int, default=12): maximum ring size to consider
    """

    if not obj.IsInRing():
        return 0
    for i in range(max_size):
        if obj.IsInRingSize(i):
            return i
    return max_size + 1


def atom_to_str(atom: 'rdkit.Chem.Atom') -> str:
    """ prepare to tokenize an atom's attributes by constructing a delimited
    string; downstream delimiter encoder assumed to handle "|" separator

    Args:
        atom (rdkit.Chem.Atom): atom to tokenize

    Returns:
        str: atom properties, delimited by "|"
    """

    props = [
        'GetChiralTag',
        'GetDegree',
        'GetExplicitValence',
        'GetFormalCharge',
        'GetHybridization',
        'GetImplicitValence',
        'GetIsAromatic',
        'GetNoImplicit',
        'GetNumExplicitHs',
        'GetNumImplicitHs',
        'GetNumRadicalElectrons',
        'GetSymbol',
        'GetTotalDegree',
        'GetTotalNumHs',
        'GetTotalValence'
    ]
    s = '|'.join([str(getattr(atom, p)()) for p in props])
    s += f'|{get_ring_size(atom)}'
    return s


def bond_to_str(bond: 'rdkit.Chem.Bond') -> str:
    """ prepare to tokenize a bond's attributes by constructing a delimited
    string; downstream delimiter encoder assumed to handle "|" separator

    Args:
        bond (rdkit.Chem.Bond): bond to tokenize

    Returns:
        str: bond properties, delimited by "|"
    """

    props = [
        'GetBondType',
        'GetIsConjugated',
        'GetStereo'
    ]
    s = '|'.join([str(getattr(bond, p)()) for p in props])
    s += f'|{bond.GetBeginAtom().GetSymbol()}'
    s += f'|{bond.GetEndAtom().GetSymbol()}'
    s += f'|{get_ring_size(bond)}'
    return s


class MoleculeEncoder(object):

    def __init__(self, smiles: List[str]):
        """ MoleculeEncoder object: given a list of SMILES strings, construct/
        train delimiter encoders to extract atom features, bond features, and
        molecule connectivity

        Args:
            smiles (List[str]): SMILES strings to consider for encoder
                construction
        """

        mols = [MolFromSmiles(smi) for smi in smiles]
        for idx, mol in enumerate(mols):
            if mol is None:
                raise ValueError(f'Unable to parse SMILES: {smiles[idx]}')

        atoms = np.concatenate([mol.GetAtoms() for mol in mols])
        atom_reprs = [atom_to_str(atom) for atom in atoms]
        bond_reprs = np.concatenate(
            [[bond_to_str(bond) for bond in atom.GetBonds()] for atom in atoms]
        )

        self._atom_encoder = DelimiterEncoder('|', atom_reprs)
        self._bond_encoder = DelimiterEncoder('|', bond_reprs)
        self._atom_dim = len(self._atom_encoder.encode(atom_reprs[0]))
        self._bond_dim = len(self._bond_encoder.encode(bond_reprs[0]))

    def encode_many(self, smiles: List[str]) -> List[
     Tuple['torch.tensor', 'torch.tensor', 'torch.tensor']]:
        """ batch encoding of SMILES strings

        Args:
            smiles (List[str]): list of SMILES strings

        Returns:
            List[Tuple[torch.tensor, torch.tensor, torch.tensor]]: List of:
                (atom encoding, bond encoding, connectivity matrix) for each
                compound
        """

        encoded_compounds = []
        for smi in smiles:
            encoded_compounds.append(self.encode(smi))
        return encoded_compounds

    def encode(self, smiles: str) -> Tuple['torch.tensor', 'torch.tensor',
                                           'torch.tensor']:
        """ encode a molecule using its SMILES string

        Args:
            smiles (str): molecule's SMILES string

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor]: (encoded atom
            features, encoded bond features, molecule connectivity matrix)
        """

        mol = rdkit.Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Unable to parse SMILES: {smiles}')
        atoms = mol.GetAtoms()

        atom_reprs = [atom_to_str(atom) for atom in atoms]
        enc_atoms = torch.stack(
            [self._atom_encoder.encode(atom) for atom in atom_reprs]
        ).type(torch.float32)

        bond_reprs = np.concatenate(
            [[bond_to_str(bond) for bond in atom.GetBonds()] for atom in atoms]
        )
        enc_bonds = torch.stack(
            [self._bond_encoder.encode(bond) for bond in bond_reprs]
        ).type(torch.float32)

        connectivity = np.zeros((2, 2 * mol.GetNumBonds()))
        bond_index = 0
        for atom in atoms:
            start_idx = atom.GetIdx()
            for bond in atom.GetBonds():
                reverse = bond.GetBeginAtomIdx() != start_idx
                if not reverse:
                    connectivity[0, bond_index] = bond.GetBeginAtomIdx()
                    connectivity[1, bond_index] = bond.GetEndAtomIdx()
                else:
                    connectivity[0, bond_index] = bond.GetEndAtomIdx()
                    connectivity[1, bond_index] = bond.GetBeginAtomIdx()
                bond_index += 1
        connectivity = torch.from_numpy(connectivity).type(torch.long)

        return (enc_atoms, enc_bonds, connectivity)
