"""End-to-end public-API smoke test (design §11.1; blueprint T1.5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from graphchem.data import MoleculeDataset, MoleculeGraph
from graphchem.datasets import load_cn
from graphchem.nn import MoleculeGCN
from graphchem.preprocessing import MoleculeEncoder


def test_load_encode_gcn_forward_backward_smoke() -> None:
    torch.manual_seed(0)

    smiles, targets = load_cn()
    smiles = smiles[:8]
    targets = targets[:8]

    encoder = MoleculeEncoder(smiles)
    encodings = encoder.encode_many(smiles)
    graphs = [
        MoleculeGraph(atoms, bonds, connectivity, targets[i])
        for i, (atoms, bonds, connectivity) in enumerate(encodings)
    ]
    dataset = MoleculeDataset(graphs)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    atom_vocab, bond_vocab = encoder.vocab_sizes
    model = MoleculeGCN(
        atom_vocab_size=atom_vocab,
        bond_vocab_size=bond_vocab,
        output_dim=1,
        embedding_dim=32,
        n_messages=1,
        n_readout=1,
        readout_dim=16,
        p_dropout=0.0,
    )
    model.train()

    batch = next(iter(loader))
    out, out_atom, out_bond = model(batch)
    assert out.shape[0] == batch.num_graphs
    assert out_atom.ndim == 2
    assert out_bond.ndim == 2

    loss = F.mse_loss(out, batch.y)
    assert torch.isfinite(loss)
    loss.backward()
