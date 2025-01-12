# test_gcn.py

import torch
from graphchem.nn import MoleculeGCN
from torch_geometric.data import Data


# Test instantiation of the MoleculeGCN class with various parameters
def test_moleculgcn_instantiation():

    model = MoleculeGCN(
        atom_vocab_size=10,
        bond_vocab_size=5,
        output_dim=2,
        embedding_dim=64,
        n_messages=3,
        n_readout=3,
        readout_dim=128,
        p_dropout=0.2
    )

    assert model._p_dropout == 0.2
    assert model._n_messages == 3


# Test the forward pass of the MoleculeGCN model
def test_moleculgcn_forward_pass():

    atom_vocab_size = 10
    bond_vocab_size = 5
    n_atoms = 3
    n_bonds = 5

    model = MoleculeGCN(
        atom_vocab_size=atom_vocab_size,
        bond_vocab_size=bond_vocab_size,
        output_dim=2,
        embedding_dim=64,
        n_messages=3,
        n_readout=3,
        readout_dim=128,
        p_dropout=0.2
    )

    n_atoms = 3

    # Create a mock input for the forward pass
    data = Data(
        x=torch.randint(0, atom_vocab_size, (n_atoms,)),
        edge_index=torch.randint(0, n_atoms, (2, n_bonds)),
        edge_attr=torch.randint(0, bond_vocab_size, (n_bonds,)),
        batch=torch.tensor([0, 0, 0], dtype=torch.long)
    )

    out, out_atom, out_bond = model.forward(data)

    assert out.shape == (1, 2)
    assert out_atom.shape == (3, 64)
    assert out_bond.shape == (5, 64)
