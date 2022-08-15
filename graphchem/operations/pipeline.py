from typing import List

import torch

import graphchem
from graphchem.data import MoleculeDataset, MoleculeGraph


def smiles_to_dataset(
    smiles: List[str], target_vals: 'torch.tensor',
    encoder: 'graphchem.preprocessing.MoleculeEncoder'
   ) -> 'graphchem.data.MoleculeDataset':
    """ smiles_to_dataset: given supplied SMILES strings, target values, and a
    graphchem.preprocessing.MoleculeEncoder, construct a
    graphchem.data.MoleculeDataset comprised of graphchem.data.MoleculeGraph
    objects

    Args:
        smiles (List[str]): SMILES strings
        target_vals ('torch.tensor'): target values
        encoder ('grapchem.preprocessing.MoleculeEncoder): encoder used for
            feature generation

    Returns:
        graphchem.data.MoleculeDataset: constructed dataset
    """

    if len(smiles) != len(target_vals):
        raise ValueError(
            'Must supply SMILES and target values of same length:' +
            f' {len(smiles)}, {len(target_vals)}'
        )
    graphs = []
    for idx, smi in enumerate(smiles):
        atom_features, bond_features, connectivity = encoder.encode(smi)
        graphs.append(MoleculeGraph(
            atom_features, bond_features, connectivity, target_vals[idx]
        ))
    return MoleculeDataset(graphs)
