import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.data as gdata

import warnings

from .utils import build_conv_layer


def _default_config():

    return {
        'n_messages': 1,
        'n_hidden': 1,
        'hidden_msg_dim': 32,
        'hidden_dim': 32,
        'dropout': 0.05
    }


class MessagePassingNet(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, task: str = 'graph',
                 config: dict = None):

        # Initialize
        super(MessagePassingNet, self).__init__()
        if not (task == 'node' or task == 'graph'):
            raise RuntimeError('{}: unknown task: {}'.format(self, task))
        self.task = task
        self._input_dim = input_dim
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

        # Construct message passing layers
        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.MFConv(
            self._input_dim, self._config['hidden_msg_dim']
        ))
        for _ in range(self._config['n_messages'] - 1):
            self.convs.append(pyg_nn.MFConv(
                self._config['hidden_msg_dim'], self._config['hidden_msg_dim']
            ))

        # Construct post-message passing layers
        self.post_conv = nn.ModuleList()
        self.post_conv.append(nn.Sequential(
            nn.Linear(self._config['hidden_msg_dim'],
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        # Feed forward
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self._config['dropout'], training=self.training)

        # If analyzing graph, perform summation over all atoms
        if self.task == 'graph':
            x = pyg_nn.global_add_pool(x, batch)

        # Perform post-message passing functions
        for layer in self.post_conv:
            x = layer(x)

        # Return atom embedding, fed-forward data
        return emb, x

    def loss(self, pred: torch.tensor, target: torch.tensor) -> torch.tensor:

        if len(target.shape) == 1:
            target = torch.reshape(target, (len(target), 1))
        return F.mse_loss(pred, target)
