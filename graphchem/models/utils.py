import torch.nn as nn
import torch_geometric.nn as pyg_nn


def build_conv_layer(input_dim: int, hidden_dim: int,
                     task: str = 'graph') -> pyg_nn.MessagePassing:

    if task == 'graph':
        return pyg_nn.GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
    else:
        return pyg_nn.GCNConv(input_dim, hidden_dim)
