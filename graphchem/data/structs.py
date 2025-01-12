from typing import Iterable, Optional

import torch
from torch_geometric.data import Data, Dataset


class MoleculeGraph(Data):
    """
    A custom graph class representing a molecular structure.

    This class extends the `Data` class from PyTorch Geometric to represent
    molecules with node attributes (atoms), edge attributes (bonds), and
    connectivity information. It also includes an optional target value.

    Attributes
    ----------
    x : torch.Tensor
        The node features (atom attributes).
    edge_index : torch.Tensor
        A 2D tensor describing the connectivity between atoms.
    edge_attr : torch.Tensor
        Edge features (bond attributes).
    y : torch.Tensor
        Target value(s) of the molecule.
    """

    def __init__(self, atom_attr: torch.Tensor,
                 bond_attr: torch.Tensor,
                 connectivity: torch.Tensor,
                 target: Optional[torch.Tensor] = None):
        """
        Initialize the MoleculeGraph object.

        Parameters
        ----------
        atom_attr : torch.Tensor
            A 2D tensor of shape (num_atoms, num_atom_features) representing
            the attributes of each atom in the molecule.
        bond_attr : torch.Tensor
            A 2D tensor of shape (num_bonds, num_bond_features) representing
            the attributes of each bond in the molecule.
        connectivity : torch.Tensor
            A 2D tensor of shape (2, num_bonds) where each column represents an
            edge (bond) between two atoms. The first row contains the source
            atom indices and the second row contains the target atom indices.
        target : Optional[torch.Tensor]
            An optional 1D or 2D tensor representing the target value(s) of the
            molecule. If not provided, it defaults to a tensor with a single
            element set to 0.0.
        """

        if target is None:
            # Set default target to a tensor with shape (1, 1) and value 0.0
            target = torch.tensor([0.0]).type(torch.float32).reshape(1, 1)
        elif len(target.shape) == 1:
            # Reshape target if it's a 1D tensor to (1, target.shape[0])
            target = target.reshape(1, -1)
        if target.shape[0] != 1:
            raise ValueError("Target tensor must have shape (1, num_targets)")

        super().__init__(
            x=atom_attr,
            edge_index=connectivity,
            edge_attr=bond_attr,
            y=target
        )


class MoleculeDataset(Dataset):
    """
    A custom dataset class for molecular graphs.

    This class extends the `Dataset` class from PyTorch Geometric to create a
    dataset of molecular graphs. Each graph in the dataset is an instance of
    `MoleculeGraph`.

    Attributes
    ----------
    _graphs : List[MoleculeGraph]
        A list containing all the molecule graphs in the dataset.
    """

    def __init__(self, graphs: Iterable[MoleculeGraph]):
        """
        Initialize the MoleculeDataset object.

        Parameters
        ----------
        graphs : Iterable[MoleculeGraph]
            An iterable of `MoleculeGraph` instances representing the
            molecules in the dataset.
        """

        super().__init__()
        self._graphs = list(graphs)

    def len(self) -> int:
        """
        Returns the number of molecules in the dataset.

        Returns
        -------
        int
            The number of molecule graphs in the dataset.
        """
        return len(self._graphs)

    def get(self, idx: int) -> MoleculeGraph:
        """
        Retrieves a molecule graph from the dataset by index.

        Returns
        -------
        MoleculeGraph
            The molecule graph at the specified index.
        """
        return self._graphs[idx]
