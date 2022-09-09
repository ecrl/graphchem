from typing import List, Tuple

import torch
import torch_geometric


class MoleculeGraph(torch_geometric.data.Data):

    def __init__(self, atom_attr: 'torch.tensor', bond_attr: 'torch.tensor',
                 connectivity: 'torch.tensor', target: 'torch.tensor' = None):
        """ MoleculeGraph object, extends torch_geometric.data.Data object; a
        singular molecule graph/data point

        Args:
            atom_attr (torch.tensor): atom features, shape (n_atoms,
                n_atom_features); dtype assumed torch.float32
            bond_attr (torch.tensor): bond features, shape (n_bonds,
                n_bond_features); dtype assumed torch.float32
            connectivity (torch.tensor): COO graph connectivity index, size
                (2, n_bonds); dtype assumed torch.long
            target (torch.tensor, default=None): target value(s), shape
                (1, n_targets); if not supplied (None), set to [0.0]; dtype
                assumed torch.float32
        """

        if target is None:
            target = torch.tensor([0.0]).type(torch.float32).reshape(1, 1)

        super(MoleculeGraph, self).__init__(
            x=atom_attr,
            edge_index=connectivity,
            edge_attr=bond_attr,
            y=target
        )


class MoleculeDataset(torch_geometric.data.Dataset):

    def __init__(self, graphs: List['MoleculeGraph']):
        """ MoleculeDataset object, extends torch_geometric.data.Dataset
        object; a torch_geometric-iterable dataset comprised of MoleculeGraph
        objects

        Args:
            graphs (List[MoleculeGraph]): list of molecule graphs
        """

        super(MoleculeDataset, self).__init__()
        self._graphs = graphs

    def len(self) -> int:
        """ torch_geometric.data.Dataset.len definition (required)

        Returns:
            int: number of molecule graphs
        """

        return len(self._graphs)

    def get(self, idx: int) -> 'MoleculeGraph':
        """ torch_geometric.data.Dataset.get definition (required)

        Args:
            idx (int): index of item

        Returns:
            MoleculeGraph: indexed item
        """

        return self._graphs[idx]
