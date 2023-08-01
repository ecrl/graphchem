r"""Encoding/tokenizing SMILES strings (preparing for graph construction)"""

import pickle
from typing import List, Tuple, Union
import rdkit
import numpy as np
import torch


def get_ring_size(obj: Union['rdkit.Chem.Atom', 'rdkit.Chem.Bond'],
                  max_size: int = 12):
    """ determine whether rdkit.Chem.Atom or rdkit.Chem.Bond is in a ring, and
    of which size

    Args:
        obj (Union[rdkit.Chem.Atom, rdkit.Chem.Bond]): atom or bond
        max_size (int): maximum ring size to consider
    """

    if not obj.IsInRing():
        return 0
    for i in range(max_size):
        if obj.IsInRingSize(i):
            return i
    return max_size + 1


def atom_to_str(atom: 'rdkit.Chem.Atom') -> str:
    """ prepare to tokenize an atom's attributes by constructing a unique
    string

    Args:
        atom (rdkit.Chem.Atom): atom to tokenize

    Returns:
        str: atom properties
    """

    return str((
        atom.GetChiralTag(),
        atom.GetDegree(),
        atom.GetExplicitValence(),
        atom.GetFormalCharge(),
        atom.GetHybridization(),
        atom.GetImplicitValence(),
        atom.GetIsAromatic(),
        atom.GetNoImplicit(),
        atom.GetNumExplicitHs(),
        atom.GetNumImplicitHs(),
        atom.GetNumRadicalElectrons(),
        atom.GetSymbol(),
        atom.GetTotalDegree(),
        atom.GetTotalNumHs(),
        atom.GetTotalValence(),
        get_ring_size(atom)
    ))


def bond_to_str(bond: 'rdkit.Chem.Bond') -> str:
    """ prepare to tokenize a bond's attributes by constructing a unique string

    Args:
        bond (rdkit.Chem.Bond): bond to tokenize

    Returns:
        str: bond properties
    """

    return str((
        bond.GetBondType(),
        bond.GetIsConjugated(),
        bond.GetStereo(),
        get_ring_size(bond),
        [sorted([bond.GetBeginAtom().GetSymbol(),
                 bond.GetEndAtom().GetSymbol()])]
    ))


class Tokenizer(object):

    def __init__(self):
        """ Tokenizer object: integer tokenizer for unique atom/bond strings
        """

        self._data = {'unk': 1}
        self.num_classes = 1
        self.train = True
        self.unknown = []

    def __call__(self, item: str) -> int:
        """ Tokenizer(): returns integer value of atom/bond string, otherwise
        'unknown', or 1; if training the tokenizer, add item to vocabulary

        Args:
            item (str): atom/bond string

        Returns:
            int: integer value of atom/bond string
        """

        try:
            return self._data[item]
        except KeyError:
            if self.train:
                self.num_classes += 1
                self._data[item] = self.num_classes
                return self(item)
            else:
                self.unknown.append(item)
                return 1

    @property
    def vocab_size(self) -> int:
        """ vocab_size: returns the total number of unique atom/bond strings
        in the tokenizer's vocabulary

        Returns:
            int: number of strings in vocabulary
        """

        return self.num_classes + 1


class MoleculeEncoder(object):

    def __init__(self, smiles: List[str]):
        """ MoleculeEncoder object: given a list of SMILES strings, construct/
        train integer tokenizers to tokenize atom/bond features, parse
        molecule connectivity

        Args:
            smiles (List[str]): SMILES strings to consider for encoder
                construction
        """

        mols = [rdkit.Chem.MolFromSmiles(smi) for smi in smiles]
        for idx, mol in enumerate(mols):
            if mol is None:
                raise ValueError(f'Unable to parse SMILES: {smiles[idx]}')

        atoms = np.concatenate([mol.GetAtoms() for mol in mols])
        atom_reprs = [atom_to_str(atom) for atom in atoms]
        bond_reprs = np.concatenate(
            [[bond_to_str(bond) for bond in atom.GetBonds()] for atom in atoms]
        )

        self._atom_tokenizer = Tokenizer()
        for rep in atom_reprs:
            self._atom_tokenizer(rep)
        self._atom_tokenizer.train = False

        self._bond_tokenizer = Tokenizer()
        for rep in bond_reprs:
            self._bond_tokenizer(rep)
        self._atom_tokenizer.train = False

    @property
    def vocab_sizes(self) -> Tuple[int]:
        """ total vocabulary/dictionary sizes for tokenizers, in form (atom
        vocab size, bond vocab size)

        Returns:
            Tuple[int]: (atom vocab size, bond vocab size)
        """

        return (self._atom_tokenizer.vocab_size,
                self._bond_tokenizer.vocab_size)

    def encode_many(self, smiles: List[str]) -> List[Tuple['torch.tensor']]:
        """ batch encoding of SMILES strings

        Args:
            smiles (List[str]): list of SMILES strings

        Returns:
            List[Tuple['torch.tensor']]: List of: (atom encoding, bond
                encoding, connectivity matrix) for each compound
        """

        encoded_compounds = []
        for smi in smiles:
            encoded_compounds.append(self.encode(smi))
        return encoded_compounds

    def encode(self, smiles: str) -> Tuple['torch.tensor']:
        """ encode a molecule using its SMILES string

        Args:
            smiles (str): molecule's SMILES string

        Returns:
            Tuple['torch.tensor']: (encoded atom features, encoded bond
                features, molecule connectivity matrix)
        """

        mol = rdkit.Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Unable to parse SMILES: {smiles}')
        atoms = mol.GetAtoms()

        atom_reprs = [atom_to_str(atom) for atom in atoms]
        enc_atoms = torch.tensor([self._atom_tokenizer(atom)
                                  for atom in atom_reprs]).type(torch.int)

        bond_reprs = np.concatenate(
            [[bond_to_str(bond) for bond in atom.GetBonds()] for atom in atoms]
        )
        enc_bonds = torch.tensor([self._bond_tokenizer(bond)
                                  for bond in bond_reprs]).type(torch.int)

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

    def save(self, filename: str) -> None:
        """ save the encoder to a file

        Args:
            filename (str): new filename/path for model

        Returns:
            None
        """

        with open(filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, filename: str) -> None:
        """ load an encoder from file (current encoder attributes, including
        pre-trained tokenizers, are overwritten)

        Args:
            filename (str): filename/path of model

        Returns:
            None
        """

        with open(filename, 'rb') as inp:
            self.__dict__.update(pickle.loads(inp).__dict__)


def load_encoder(filename: str) -> MoleculeEncoder:
    """ loads a pre-saved `MoleculeEncoder` object

    Args:
        filename (str): filename/path of saved encoder

    Returns:
        MoleculeEncoder: loaded encoder object
    """

    with open(filename, 'rb') as inp:
        encoder = pickle.loads(inp)
    return encoder
