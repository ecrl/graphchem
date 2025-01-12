import pytest
import torch

from graphchem.data.structs import MoleculeGraph, MoleculeDataset


# Test cases for MoleculeGraph initialization
def test_moleculargraph_initialization():

    # Test initialization with no target
    atom_attr = torch.rand((10, 5))  # 10 atoms, each with 5 features
    bond_attr = torch.rand((20, 4))  # 20 bonds, each with 4 features
    connectivity = torch.randint(0, 10, (2, 20))

    mol_graph_no_target = MoleculeGraph(atom_attr, bond_attr, connectivity)
    assert mol_graph_no_target.x.shape == atom_attr.shape
    assert mol_graph_no_target.edge_index.shape == connectivity.shape
    assert mol_graph_no_target.edge_attr.shape == bond_attr.shape
    assert mol_graph_no_target.y.shape == (1, 1) and torch.allclose(
        mol_graph_no_target.y, torch.tensor([[0.]])
    )

    # Test initialization with a 1D target tensor
    target = torch.tensor([2.5])
    mol_graph_1d_target = MoleculeGraph(atom_attr, bond_attr, connectivity,
                                        target)
    assert mol_graph_1d_target.y.shape == (1, 1) and torch.allclose(
        mol_graph_1d_target.y, torch.tensor([[2.5]])
    )

    # Test initialization with a 2D target tensor
    target = torch.rand((1, 3))
    mol_graph_2d_target = MoleculeGraph(atom_attr, bond_attr, connectivity,
                                        target)
    assert mol_graph_2d_target.y.shape == target.shape

    # Test initialization with a 1D target tensor, multiple values
    target = torch.rand(3)
    mol_graph_1d_multi_target = MoleculeGraph(
        atom_attr, bond_attr, connectivity, target
    )
    assert mol_graph_1d_multi_target.y.shape == (1, 3)


# Test cases for MoleculeGraph with invalid target shape
def test_moleculargraph_invalid_target():

    # Test initialization with a 2D target tensor that cannot be reshaped to
    # (1, num_targets)
    target = torch.rand((2, 3))  # This will raise an error
    with pytest.raises(ValueError):
        MoleculeGraph(torch.rand((10, 5)), torch.rand((20, 4)),
                      torch.randint(0, 10, (2, 20)), target)


# Test cases for MoleculeDataset initialization
def test_moleculardataset_initialization():

    # Test initialization with a list of MoleculeGraphs
    atom_attr = torch.rand((10, 5))  # 10 atoms, each with 5 features
    bond_attr = torch.rand((20, 4))  # 20 bonds, each with 4 features
    connectivity = torch.randint(0, 10, (2, 20))

    graphs = [MoleculeGraph(atom_attr, bond_attr, connectivity)
              for _ in range(5)]
    dataset = MoleculeDataset(graphs)

    assert len(dataset) == 5
    assert isinstance(dataset[0], MoleculeGraph)


# Test cases for MoleculeDataset lengths
def test_moleculardataset_len():

    # Test len method with an empty dataset
    dataset = MoleculeDataset([])
    assert len(dataset) == 0

    # Test len method with a non-empty dataset
    atom_attr = torch.rand((10, 5))  # 10 atoms, each with 5 features
    bond_attr = torch.rand((20, 4))  # 20 bonds, each with 4 features
    connectivity = torch.randint(0, 10, (2, 20))

    graphs = [MoleculeGraph(atom_attr, bond_attr, connectivity)
              for _ in range(5)]
    dataset = MoleculeDataset(graphs)
    assert len(dataset) == 5


# Test cases for MoleculeDataset indexing
def test_moleculardataset_get():

    # Test get method with a non-empty dataset
    atom_attr = torch.rand((10, 5))  # 10 atoms, each with 5 features
    bond_attr = torch.rand((20, 4))  # 20 bonds, each with 4 features
    connectivity = torch.randint(0, 10, (2, 20))

    graphs = [MoleculeGraph(atom_attr, bond_attr, connectivity)
              for _ in range(5)]
    dataset = MoleculeDataset(graphs)

    assert isinstance(dataset[0], MoleculeGraph)
    with pytest.raises(IndexError):
        _ = dataset[5]  # Trying to access an index out of bounds
