from torch_geometric.data import Data, DataLoader
from torch import tensor, float as tfloat


def dataloader_from_mols(mols: list, targets: list = None,
                         batch_size: int = 32) -> DataLoader:

    if targets is None:
        graphs = [Data(
            x=tensor(mols[idx].atom_features),
            edge_index=tensor(mols[idx].edge_connectivity)
        ) for idx in range(len(mols))]
    else:
        if len(mols) != len(targets):
            raise ValueError('Must supply same number of molecules and '
                             'targets: {}, {}'.format(len(mols), len(targets)))
        graphs = [Data(
                    x=tensor(mols[idx].atom_features),
                    edge_index=tensor(mols[idx].edge_connectivity),
                    y=tensor(targets[idx])
                )for idx in range(len(mols))]
    return DataLoader(graphs, batch_size=batch_size, shuffle=True)
