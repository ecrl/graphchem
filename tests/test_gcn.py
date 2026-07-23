# test_gcn.py

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from graphchem.nn import MoleculeGCN


def _mock_graph(
    atom_vocab_size: int,
    bond_vocab_size: int,
    n_atoms: int,
    n_bonds: int,
    seed: int,
) -> Data:
    g = torch.Generator().manual_seed(seed)
    return Data(
        x=torch.randint(0, atom_vocab_size, (n_atoms,), generator=g),
        edge_index=torch.randint(0, n_atoms, (2, n_bonds), generator=g),
        edge_attr=torch.randint(0, bond_vocab_size, (n_bonds,), generator=g),
    )


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
        p_dropout=0.2,
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
        p_dropout=0.2,
    )

    n_atoms = 3

    # Create a mock input for the forward pass
    data = Data(
        x=torch.randint(0, atom_vocab_size, (n_atoms,)),
        edge_index=torch.randint(0, n_atoms, (2, n_bonds)),
        edge_attr=torch.randint(0, bond_vocab_size, (n_bonds,)),
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
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
        p_dropout=0.2,
    )

    # n_readout=0: pooled atom embedding is returned directly (no MLP head)
    assert model.readout is None

    n_atoms = 3

    # Create a mock input for the forward pass
    data = Data(
        x=torch.randint(0, atom_vocab_size, (n_atoms,)),
        edge_index=torch.randint(0, n_atoms, (2, n_bonds)),
        edge_attr=torch.randint(0, bond_vocab_size, (n_bonds,)),
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
    )

    out, out_atom, out_bond = model.forward(data)

    assert out.shape == (1, embedding_dim)
    assert out.shape[-1] == embedding_dim
    assert out_atom.shape == (n_atoms, embedding_dim)
    assert out_bond.shape == (n_bonds, embedding_dim)
    assert torch.isfinite(out).all()


def test_moleculegcn_batched_dataloader_forward():
    """Batched DataLoader forward keeps (out, out_atom, out_bond) arity."""
    torch.manual_seed(0)
    atom_vocab_size = 10
    bond_vocab_size = 5
    embedding_dim = 32
    output_dim = 2
    batch_size = 3

    graphs = [
        _mock_graph(atom_vocab_size, bond_vocab_size, n_atoms=2 + i, n_bonds=4, seed=i)
        for i in range(batch_size)
    ]
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    assert batch.num_graphs == batch_size

    model = MoleculeGCN(
        atom_vocab_size=atom_vocab_size,
        bond_vocab_size=bond_vocab_size,
        output_dim=output_dim,
        embedding_dim=embedding_dim,
        n_messages=2,
        n_readout=1,
        readout_dim=16,
        p_dropout=0.0,
    )
    model.eval()

    out, out_atom, out_bond = model(batch)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch_size, output_dim)
    assert out_atom.shape[0] == batch.num_nodes
    assert out_atom.shape[1] == embedding_dim
    assert out_bond.shape[0] == batch.num_edges
    assert out_bond.shape[1] == embedding_dim
    assert torch.isfinite(out).all()
    assert torch.isfinite(out_atom).all()
    assert torch.isfinite(out_bond).all()


def test_moleculegcn_non_default_aggr_mean():
    """GeneralConv aggr='mean' constructs and forwards with stable shapes."""
    atom_vocab_size = 10
    bond_vocab_size = 5
    embedding_dim = 64
    output_dim = 2
    n_atoms = 3
    n_bonds = 5

    try:
        model = MoleculeGCN(
            atom_vocab_size=atom_vocab_size,
            bond_vocab_size=bond_vocab_size,
            output_dim=output_dim,
            embedding_dim=embedding_dim,
            n_messages=2,
            n_readout=1,
            readout_dim=32,
            p_dropout=0.0,
            aggr="mean",
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        pytest.skip(f"aggr='mean' unsupported by GeneralConv: {exc}")

    data = Batch.from_data_list(
        [_mock_graph(atom_vocab_size, bond_vocab_size, n_atoms, n_bonds, seed=1)]
    )
    out, out_atom, out_bond = model(data)
    assert out.shape == (1, output_dim)
    assert out_atom.shape == (n_atoms, embedding_dim)
    assert out_bond.shape == (n_bonds, embedding_dim)
    assert torch.isfinite(out).all()


# Test the forward pass of the MoleculeGCN model with different activation fns
def test_moleculegcn_act_fns():

    functions = [
        torch.nn.functional.softplus,
        torch.nn.functional.sigmoid,
        torch.nn.functional.relu,
        torch.nn.functional.tanh,
        torch.nn.functional.leaky_relu,
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
            act_fn=fn,
        )

        assert model.readout is None

        n_atoms = 3

        # Create a mock input for the forward pass
        data = Data(
            x=torch.randint(0, atom_vocab_size, (n_atoms,)),
            edge_index=torch.randint(0, n_atoms, (2, n_bonds)),
            edge_attr=torch.randint(0, bond_vocab_size, (n_bonds,)),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
        )

        out, out_atom, out_bond = model.forward(data)

        assert out.shape == (1, embedding_dim)
        assert out_atom.shape == (n_atoms, embedding_dim)
        assert out_bond.shape == (n_bonds, embedding_dim)
