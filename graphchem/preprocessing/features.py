import pickle
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import rdkit
import torch


def get_ring_size(
        obj: Union["rdkit.Chem.Atom", "rdkit.Chem.Bond"],
        max_size: Optional[int] = 12
     ) -> int:
    """
    Determine the size of the smallest ring that an atom or bond is part of.

    Parameters
    ----------
    obj : Union[rdkit.Chem.Atom, rdkit.Chem.Bond]
        An RDKit Atom or Bond object to check for ring membership.
    max_size : Optional[int], default 12
        The maximum size of the ring to consider. If no ring is found with a
        size less than or equal to `max_size`, this value will be returned.

    Returns
    -------
    int
        The size of the smallest ring that the atom or bond is part of, or
        `max_size` if no smaller ring is found.
    """
    if not obj.IsInRing():
        return 0
    for i in range(max_size):
        if obj.IsInRingSize(i):
            return i
    return max_size


def atom_to_str(atom: "rdkit.Chem.Atom") -> str:
    """
    Convert an RDKit Atom object to a string representation.

    The string representation includes various properties of the atom,
    such as chiral tag, degree, explicit valence, formal charge, hybridization,
    implicit valence, aromaticity, number of implicit hydrogen atoms, and more.

    Parameters
    ----------
    atom : rdkit.Chem.Atom
        An RDKit Atom object representing a single atom in a molecule.

    Returns
    -------
    str
        A string representation of the atom, including its properties.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C1=CC=CC=C1')
    >>> atom = mol.GetAtomWithIdx(0)
    >>> atom_to_str(atom)
    '(CHIRAL_NONE, 3, 4, 0, <Hybridization.SP2: 6>, 0, True, False, 0, 0, 0,
     'C', 3, 1, 4, 5)'
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


def bond_to_str(bond: "rdkit.Chem.Bond") -> str:
    """
    Convert an RDKit Bond object to a string representation.

    The string representation includes various properties of the bond,
    including bond type, conjugation, stereochemistry, ring size, and
    connected atom symbols.

    Parameters
    ----------
    bond : rdkit.Chem.Bond
        An RDKit Bond object representing a single bond in a molecule.

    Returns
    -------
    str
        A string representation of the bond, including its properties.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C=C')
    >>> bond = mol.GetBondWithIndices(0, 1)
    >>> bond_to_str(bond)
    "(DOUBLE, False, NONE, None, ['C', 'C'])"
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
    """
    A simple tokenizer that assigns a unique integer to each token (word) in
    the input data. If the tokenizer is in training mode, it will add new
    tokens to the vocabulary. Otherwise, it will return the integer
    corresponding to 'unk' for unknown tokens.

    Attributes
    ----------
    _data : dict
        A dictionary mapping each token to a unique integer. Initialized with
        {"unk": 1}.
    num_classes : int
        The number of unique classes (tokens) in the vocabulary, including
        'unk'.
    train : bool
        A flag indicating whether the tokenizer is in training mode.
    unknown : list
        A list to store tokens that were encountered during inference but are
        not in the vocabulary.
    """

    def __init__(self):
        """
        Initialize the Tokenizer with default values.
        """
        self._data = {"unk": 1}
        self.num_classes = 1
        self.train = True
        self.unknown = []

    def __call__(self, item: str) -> int:
        """
        Tokenizes a given string by returning its corresponding integer from
        the vocabulary.

        Parameters
        ----------
        item : str
            The token (word) to be tokenized.

        Returns
        -------
        int
            The unique integer assigned to the token. If the token is not in
            the vocabulary and the tokenizer is in training mode, it will add
            the token and return its corresponding integer. Otherwise, it
            returns 1, which corresponds to 'unk'.
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
        """
        Returns the size of the vocabulary, which is the number of unique
        tokens plus one.

        Returns
        -------
        int
            The total number of classes (tokens) in the vocabulary plus one.
        """
        return self.num_classes + 1


class MoleculeEncoder(object):
    """
    A class to encode molecular SMILES strings into numerical (integer)
    representations using tokenized atom and bond information.

    Attributes
    ----------
    _atom_tokenizer : Tokenizer
        integer Tokenizer for atom representations.
    _bond_tokenizer : Tokenizer
        integer Tokenizer for bond representations.
    """

    def __init__(self, smiles: List[str]):
        """
        Initializes the MoleculeEncoder with a list of SMILES strings and
        creates/trains integer tokenizers for atoms and bonds.

        Parameters
        ----------
        smiles : List[str]
            A list of SMILES strings representing molecules used for tokenizer
            creation/training.

        Raises
        ------
        ValueError
            If any provided SMILES string cannot be parsed by RDKit.
        """
        mols = [rdkit.Chem.MolFromSmiles(smi) for smi in smiles]
        for idx, mol in enumerate(mols):
            if mol is None:
                raise ValueError(f"Unable to parse SMILES: {smiles[idx]}")

        atoms = np.concatenate([mol.GetAtoms() for mol in mols])
        atom_reprs = [atom_to_str(atom) for atom in atoms]
        bond_reprs = np.concatenate([
            [bond_to_str(bond) for bond in atom.GetBonds()]
            for atom in atoms
        ])

        self._atom_tokenizer = Tokenizer()
        for rep in atom_reprs:
            self._atom_tokenizer(rep)
        self._atom_tokenizer.train = False

        self._bond_tokenizer = Tokenizer()
        for rep in bond_reprs:
            self._bond_tokenizer(rep)
        self._bond_tokenizer.train = False

    @property
    def vocab_sizes(self) -> Tuple[int, int]:
        """
        Returns the vocabulary sizes of the atom and bond tokenizers.

        Returns
        -------
        Tuple[int, int]
            A tuple containing two integers representing the sizes of the atom
            and bond tokenizers' vocabularies respectively.
        """
        return (
            self._atom_tokenizer.vocab_size,
            self._bond_tokenizer.vocab_size
        )

    def encode(
            self,
            smiles: str
         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes a single SMILES string into three tensors representing atoms,
        bonds, and connectivity.

        Parameters
        ----------
        smiles : str
            A SMILES string representing the molecule to be encoded.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
                - A tensor of atom encodings, shape (n_atoms,).
                - A tensor of bond encodings, shape (n_bonds,).
                - A connectivity matrix as a tensor, shape (2, n_bonds).

        Raises
        ------
        ValueError
            If the provided SMILES string cannot be parsed by RDKit.
        """
        mol = rdkit.Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Unable to parse SMILES string: {smiles}")
        atoms = mol.GetAtoms()

        atom_reprs = [atom_to_str(atom) for atom in atoms]
        enc_atoms = torch.tensor([
            self._atom_tokenizer(atom) for atom in atom_reprs
        ]).type(torch.int)

        bond_reprs = np.concatenate([
            [bond_to_str(bond) for bond in atom.GetBonds()]
            for atom in atoms
        ])
        enc_bonds = torch.tensor([
            self._bond_tokenizer(bond) for bond in bond_reprs
        ]).type(torch.int)

        connectivity = np.zeros((2, 2 * mol.GetNumBonds()))
        bond_index = 0
        for atom in atoms:
            start_idx = atom.GetIdx()
            for bond in atom.GetBonds():
                if bond.GetBeginAtomIdx() == start_idx:
                    connectivity[0, bond_index] = bond.GetBeginAtomIdx()
                    connectivity[1, bond_index] = bond.GetEndAtomIdx()
                else:
                    connectivity[0, bond_index] = bond.GetEndAtomIdx()
                    connectivity[1, bond_index] = bond.GetBeginAtomIdx()
                bond_index += 1
        connectivity = torch.tensor(connectivity).type(torch.long)

        return enc_atoms, enc_bonds, connectivity

    def encode_many(
            self,
            smiles: Iterable[str]
         ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Encodes a list of SMILES strings into tensors representing atoms,
        bonds, and connectivities.

        Parameters
        ----------
        smiles : Iterable[str]
            An iterable collection of SMILES strings representing molecules to
            be encoded.

        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            A list containing tuples with three elements:
                - A tensor of atom encodings, shape (n_atoms,).
                - A tensor of bond encodings, shape (n_bonds,).
                - A connectivity matrix as a tensor, shape (2, n_bonds).

        Raises
        ------
        ValueError
            If any provided SMILES string cannot be parsed by RDKit.
        """
        encodings = []
        for smi in smiles:
            encodings.append(self.encode(smi))
        return encodings

    def save(self, filename: str) -> None:
        """
        Save the encoder to a file.

        Parameters
        ----------
        filename : str
            filename/path to save the encoder to.
        """
        with open(filename, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


def load_encoder(filename: str) -> MoleculeEncoder:
    """
    Loads a pre-saved `MoleculeEncoder` object from a file.

    Parameters
    ----------
    filename : str
        The path to the saved encoder file.

    Returns
    -------
    MoleculeEncoder
        The loaded `MoleculeEncoder` object.
    """
    with open(filename, "rb") as inp:
        encoder = pickle.load(inp)
    return encoder
