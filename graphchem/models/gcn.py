import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.data as gdata
from torch_scatter import scatter_add

import warnings


def _default_config():

    return {
        'n_messages': 1,
        'n_hidden': 1,
        'hidden_msg_dim': 32,
        'hidden_dim': 32,
        'dropout': 0.05
    }


class CompoundGCN(nn.Module):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int,
                 config: dict = None):

        # Initialize
        super(CompoundGCN, self).__init__()
        self._node_dim = node_dim
        self._edge_dim = edge_dim
        self._output_dim = output_dim

        # Create model config if none supplied
        if config is None:
            config = _default_config()
        # Config supplied, check for variables and variable values
        else:
            df = _default_config()
            for key in df.keys():
                try:
                    _val = config[key]
                except KeyError:
                    config[key] = df[key]
                    warnings.warn(
                        '{} config value not found: default value set, {}'
                        .format(key, df[key])
                    )
            if config['n_messages'] < 1:
                raise ValueError('config[\'n_messages\') < 1: {}'.format(
                                 config['n_messages']))
            elif config['n_hidden'] < 1:
                raise ValueError('config[\'n_hidden\' < 1: {}'.format(
                                 config['n_hidden']))
            elif config['hidden_dim'] < 1:
                raise ValueError('config[\'hidden_dim\'] < 1: {}'.format(
                                 config['hidden_dim']))
            elif config['hidden_msg_dim'] < 1:
                raise ValueError('config[\'hidden_msg_dim\'] < 1: {}'.format(
                                 config['hidden_msg_dim']))
            elif config['dropout'] < 0.0 or config['dropout'] > 1.0:
                raise ValueError('config[\'dropout\'] invalid: {}'.format(
                                 config['dropout']))
        self._config = config

    def construct(self):

        # Construct message passing layers for node network
        self.node_convs = nn.ModuleList()
        self.node_convs.append(pyg_nn.MFConv(
            self._node_dim, self._config['hidden_msg_dim']
        ))
        for _ in range(self._config['n_messages'] - 1):
            self.node_convs.append(pyg_nn.MFConv(
                self._config['hidden_msg_dim'], self._config['hidden_msg_dim']
            ))

        # Construct message passing layers for edge network
        self.edge_convs = nn.ModuleList()
        self.edge_convs.append(pyg_nn.EdgeConv(nn.Sequential(
            nn.Linear(2 * self._edge_dim, self._config['hidden_msg_dim'])
        )))
        for _ in range(self._config['n_messages'] - 1):
            self.edge_convs.append(pyg_nn.EdgeConv(nn.Sequential(
                nn.Linear(2 * self._config['hidden_msg_dim'],
                          self._config['hidden_msg_dim'])
            )))

        # Construct post-message passing layers
        self.post_conv = nn.ModuleList()
        self.post_conv.append(nn.Sequential(
            nn.Linear(2 * self._config['hidden_msg_dim'],
                      self._config['hidden_dim']),
            nn.Dropout(self._config['dropout'])
        ))
        for _ in range(self._config['n_hidden'] - 1):
            self.post_conv.append(nn.Sequential(
                nn.Linear(self._config['hidden_dim'],
                          self._config['hidden_dim']),
                nn.Dropout(self._config['dropout'])
            ))
        self.post_conv.append(nn.Sequential(
            nn.Linear(self._config['hidden_dim'], self._output_dim)
        ))

    def forward(self, data: gdata.Data) -> tuple:

        # Get batch
        x, edge_attr, edge_index, batch = data.x, data.edge_attr,\
            data.edge_index, data.batch
        row, col = edge_index
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        # Feed forward, nodes
        for i in range(len(self.node_convs)):
            x = self.node_convs[i](x, edge_index)
            emb_node = x
            x = F.relu(x)
            x = F.dropout(x, p=self._config['dropout'], training=self.training)

        # Feed forward, edges
        for i in range(len(self.edge_convs)):
            edge_attr = self.edge_convs[i](edge_attr, edge_index)
            emb_edge = edge_attr
            edge_attr = F.relu(edge_attr)
            edge_attr = F.dropout(edge_attr, p=self._config['dropout'],
                                  training=self.training)

        # Concatenate node network and edge network output tensors
        out = torch.cat([x[row], edge_attr[col]], dim=1)

        # Perform scatter add, reshape to original node dimensionality
        out = scatter_add(out, col, dim=0, dim_size=x.size(0))

        # Perform summation over all nodes w.r.t. current batch
        out = pyg_nn.global_add_pool(out, batch)

        # Perform post-message passing feed forward operations
        for layer in self.post_conv:
            out = layer(out)

        # Return fed-forward data, node embedding, edge embedding
        return out, emb_node, emb_edge

    def loss(self, pred: torch.tensor, target: torch.tensor) -> torch.tensor:

        if len(target.shape) == 1:
            target = torch.reshape(target, (len(target), 1))
        return F.mse_loss(pred, target)
