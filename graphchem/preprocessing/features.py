import torch
from torchnlp.encoders.text import DelimiterEncoder
import numpy as np
from rdkit import Chem

from .smiles import parse_smiles


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

    def encode(self, smiles: str) -> tuple:

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Unable to parse SMILES: {}'.format(smiles))
        atoms = mol.GetAtoms()
        atom_reprs = [atom_to_string(a) for a in atoms]
        bond_reprs = np.concatenate(
            [[bond_to_string(b) for b in a.GetBonds()] for a in atoms]
        )
        enc_atoms = torch.stack([self._atom_encoder.encode(a)
                                 for a in atom_reprs])
        enc_bonds = torch.stack([self._bond_encoder.encode(b)
                                 for b in bond_reprs])
        enc_atoms_pad = torch.stack([
            torch.cat((enc, torch.zeros(self.ATOM_DIM - len(enc))))
            for enc in enc_atoms
        ]).type(torch.float)
        enc_bonds_pad = torch.stack([
            torch.cat((enc, torch.zeros(self.BOND_DIM - len(enc))))
            for enc in enc_bonds
        ]).type(torch.float)
        return (enc_atoms_pad, enc_bonds_pad)

    @staticmethod
    def connectivity(smiles: str) -> torch.tensor:

        atoms = parse_smiles(smiles)
        return torch.from_numpy(
            np.concatenate([a.connectivity for a in atoms], axis=1)
        ).type(torch.long)
