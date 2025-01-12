# test_gcn.py

import torch
from graphchem.nn import MoleculeGCN
from torch_geometric.data import Data


# Test instantiation of the MoleculeGCN class with various parameters
def test_moleculegcn_instantiation():

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
    assert model.readout is not None


# Test the forward pass of the MoleculeGCN model
def test_moleculegcn_forward_pass():

    atom_vocab_size = 10
    bond_vocab_size = 5
    embedding_dim = 64
    output_dim = 2
    n_atoms = 3
    n_bonds = 5

    model = MoleculeGCN(
        atom_vocab_size=atom_vocab_size,
        bond_vocab_size=bond_vocab_size,
        output_dim=output_dim,
        embedding_dim=embedding_dim,
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

    assert out.shape == (1, output_dim)
    assert out_atom.shape == (n_atoms, embedding_dim)
    assert out_bond.shape == (n_bonds, embedding_dim)


# Test the forward pass of the MoleculeGCN model without readout layers
def test_moleculegcn_no_readout():

    atom_vocab_size = 10
    bond_vocab_size = 5
    embedding_dim = 64
    n_atoms = 3
    n_bonds = 5

    model = MoleculeGCN(
        atom_vocab_size=atom_vocab_size,
        bond_vocab_size=bond_vocab_size,
        output_dim=None,
        embedding_dim=embedding_dim,
        n_messages=3,
        n_readout=0,
        readout_dim=128,
        p_dropout=0.2
    )

    assert model.readout is None

    n_atoms = 3

    # Create a mock input for the forward pass
    data = Data(
        x=torch.randint(0, atom_vocab_size, (n_atoms,)),
        edge_index=torch.randint(0, n_atoms, (2, n_bonds)),
        edge_attr=torch.randint(0, bond_vocab_size, (n_bonds,)),
        batch=torch.tensor([0, 0, 0], dtype=torch.long)
    )

    out, out_atom, out_bond = model.forward(data)

    assert out.shape == (1, embedding_dim)
    assert out_atom.shape == (n_atoms, embedding_dim)
    assert out_bond.shape == (n_bonds, embedding_dim)


# Test the forward pass of the MoleculeGCN model with different activation fns
def test_moleculegcn_act_fns():

    functions = [
        torch.nn.functional.softplus,
        torch.nn.functional.sigmoid,
        torch.nn.functional.relu,
        torch.nn.functional.tanh,
        torch.nn.functional.leaky_relu
    ]

    for fn in functions:

        atom_vocab_size = 10
        bond_vocab_size = 5
        embedding_dim = 64
        n_atoms = 3
        n_bonds = 5

        model = MoleculeGCN(
            atom_vocab_size=atom_vocab_size,
            bond_vocab_size=bond_vocab_size,
            output_dim=None,
            embedding_dim=embedding_dim,
            n_messages=3,
            n_readout=0,
            readout_dim=128,
            p_dropout=0.2,
            act_fn=fn
        )

        assert model.readout is None

        n_atoms = 3

        # Create a mock input for the forward pass
        data = Data(
            x=torch.randint(0, atom_vocab_size, (n_atoms,)),
            edge_index=torch.randint(0, n_atoms, (2, n_bonds)),
            edge_attr=torch.randint(0, bond_vocab_size, (n_bonds,)),
            batch=torch.tensor([0, 0, 0], dtype=torch.long)
        )

        out, out_atom, out_bond = model.forward(data)

        assert out.shape == (1, embedding_dim)
        assert out_atom.shape == (n_atoms, embedding_dim)
        assert out_bond.shape == (n_bonds, embedding_dim)
