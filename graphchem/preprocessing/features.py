import torch
from torchnlp.encoders.text import DelimiterEncoder
import numpy as np
from rdkit import Chem
from typing import Tuple


def atom_to_string(atom: Chem.Atom) -> str:

    return '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}'.format(
        atom.GetChiralTag(),
        atom.GetDegree(),
        atom.GetExplicitValence(),
        atom.GetFormalCharge(),
        atom.GetHybridization(),
        atom.GetIsAromatic(),
        atom.GetMass(),
        atom.GetNumExplicitHs(),
        atom.GetNumImplicitHs(),
        atom.GetNumRadicalElectrons(),
        atom.GetTotalDegree(),
        atom.GetTotalNumHs(),
        atom.GetTotalValence(),
        atom.IsInRing()
    )


def bond_to_string(bond: Chem.Bond) -> str:

    return '{}|{}|{}|{}|{}|{}|{}'.format(
        bond.GetBondType(),
        bond.GetIsAromatic(),
        bond.GetIsConjugated(),
        bond.GetStereo(),
        bond.IsInRing(),
        bond.GetBeginAtom().GetSymbol(),
        bond.GetEndAtom().GetSymbol()
    )


class CompoundEncoder(object):

    def __init__(self, smiles: list):
        """
        CompoundEncoder(object): given a list of SMILES strings, tokenizes and
        encodes all atoms and bonds for all SMILES strings, calculate
        connectivity

        Utilizes RDKit for atom and bond feature calculation (e.g. atom mass,
        bond degree), and torchnlp.encoders.text.DelimiterEncoder for encoding/
        vectorizing SMILES strings

        n_features per atom and bond will be, at a minimum, 32; higher
        n_features is possible given large variance in the supplied SMILES
        strings, or just a lot of SMILES strings
        """

        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        for idx, m in enumerate(mols):
            if m is None:
                raise ValueError('Unable to parse SMILES: {}'.format(
                    smiles[idx]
                ))
        atoms = np.concatenate([mol.GetAtoms() for mol in mols])
        atom_reprs = [atom_to_string(a) for a in atoms]
        bond_reprs = np.concatenate(
            [[bond_to_string(b) for b in a.GetBonds()] for a in atoms]
        )
        self._atom_encoder = DelimiterEncoder('|', atom_reprs)
        self._bond_encoder = DelimiterEncoder('|', bond_reprs)
        self.ATOM_DIM = max(32, len(self._atom_encoder.encode(atom_reprs[0])))
        self.BOND_DIM = max(32, len(self._bond_encoder.encode(bond_reprs[0])))

    def encode(self, smiles: str) -> Tuple['torch.tensor', 'torch.tensor',
                                           'torch.tensor']:
        """
        Given the SMILES strings used to initialize CompoundEncoder and
        subsequently used to train atom and bond encoders, encode another
        SMILES string with these encoders

        Args:
            smiles (str): SMILES string to encode

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor]: encoded atoms of
                shape (n_atoms, n_atom_features), encoded bonds of shape
                (n_bonds, n_bond_features), connectivity of shape (2, n_bonds)
                i.e. COO graph connectivity format
        """

        # Generate mol. with RDKit, get atoms
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Unable to parse SMILES: {}'.format(smiles))
        atoms = mol.GetAtoms()

        # Encode atoms
        atom_reprs = [atom_to_string(a) for a in atoms]
        enc_atoms = torch.stack([self._atom_encoder.encode(a)
                                 for a in atom_reprs])
        enc_atoms_pad = torch.stack([
            torch.cat((enc, torch.zeros(self.ATOM_DIM - len(enc))))
            for enc in enc_atoms
        ]).type(torch.float)

        # Encode bonds
        bond_reprs = np.concatenate(
            [[bond_to_string(b) for b in a.GetBonds()] for a in atoms]
        )
        enc_bonds = torch.stack([self._bond_encoder.encode(b)
                                 for b in bond_reprs])
        enc_bonds_pad = torch.stack([
            torch.cat((enc, torch.zeros(self.BOND_DIM - len(enc))))
            for enc in enc_bonds
        ]).type(torch.float)

        # Determine COO graph connectivity
        connectivity = np.zeros((2, 2 * mol.GetNumBonds()))
        bond_index = 0
        for atom in mol.GetAtoms():
            for bond in atom.GetBonds():
                connectivity[0, bond_index] = bond.GetBeginAtomIdx()
                connectivity[1, bond_index] = bond.GetEndAtomIdx()
                bond_index += 1
        connectivity = torch.from_numpy(connectivity).type(torch.long)

        return (enc_atoms_pad, enc_bonds_pad, connectivity)
