import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.data as gdata
from torch_scatter import scatter_add
from typing import Tuple


class CompoundGCN(nn.Module):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int,
                 n_messages: int = 2, n_hidden: int = 1, hidden_dim: int = 32,
                 dropout: float = 0.0):
        """
        CompoundGCN(torch.nn.Module): Combines torch_geometric.nn.MFConv and
        torch_geometric.nn.EdgeConv to perform `n_messages` graph node and
        graph edge convolutions; on graph readout, node and edge features
        concatenated, scatter added, global added w.r.t. current batch, passed
        through fully connected NN

        TODO: Write out math

        Args:
            node_dim (int): n_features for node feature vectors
            edge_dim (int): n_features for edge feature vectors
            output_dim (int): n_features for target value vectors
            n_messages (int, optional): number of message-passing ops to
                perform for nodes and edges, default = 2
            n_hidden (int, optional): number of sequential, feed-forward
                layers after graph node/edge readout/scatter mean, default = 1
            hidden_dim (int, optional): number of neurons in feed-forward
                readout net's hidden layers, default = 32
            dropout (float, optional): neuron dropout rate, default = 0.0
        """

        super(CompoundGCN, self).__init__()
        self._dropout = dropout
        self._n_messages = n_messages

        self.lin0 = nn.Linear(node_dim, node_dim)

        # Construct message passing layers for node, edge networks
        self.node_conv = pyg_nn.MFConv(node_dim, node_dim)
        self.edge_conv = pyg_nn.EdgeConv(nn.Sequential(
            nn.Linear(2 * edge_dim, edge_dim)
        ))
        self.node_gru = nn.GRU(node_dim, node_dim)

        # Construct post-message passing layers
        self.post_conv = nn.ModuleList()
        self.post_conv.append(nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim)
        ))
        for _ in range(n_hidden - 1):
            self.post_conv.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim)
            ))
        self.post_conv.append(nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        ))

    def forward(self, data: 'torch_geometric.data.Data') -> Tuple[
     'torch.tensor', 'torch.tensor', 'torch.tensor']:
        """
        torch.nn.module forward operation

        Args:
            data (torch_geometric.data.Data): data to be fed forward; must have
                node attributes, edge attributes, edge index defined

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor]: (GCN output, node
                embeddings, edge embeddings)
        """

        # Get batch
        x, edge_attr, edge_index, batch = data.x, data.edge_attr,\
            data.edge_index, data.batch
        row, col = edge_index
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        # Feed forward, node and edge messages
        for i in range(self._n_messages):
            m = F.relu(self.node_conv(out, edge_index))
            emb_node = m
            m = F.dropout(m, p=self._dropout, training=self.training)
            out, h = self.node_gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            edge_attr = self.edge_conv(edge_attr, edge_index)
            emb_edge = edge_attr
            edge_attr = F.relu(edge_attr)
            edge_attr = F.dropout(edge_attr, p=self._dropout,
                                  training=self.training)

        # Concatenate node network and edge network output tensors
        out = torch.cat([out[row], edge_attr[col]], dim=1)

        # Perform scatter add, reshape to original node dimensionality
        out = scatter_add(out, col, dim=0, dim_size=x.size(0))

        # Perform summation over all nodes w.r.t. current batch
        out = pyg_nn.global_add_pool(out, batch)

        # Perform post-message passing feed forward operations
        for layer in self.post_conv[:-1]:
            out = layer(out)
            out = F.relu(out)
            out = F.dropout(out, p=self._dropout, training=self.training)
        out = self.post_conv[-1](out)

        # Return fed-forward data, node embedding, edge embedding
        return out, emb_node, emb_edge

    def loss(self, pred: 'torch.tensor',
             target: 'torch.tensor') -> 'torch.tensor':
        """
        Computes MSE loss between two tensors of equal size

        Args:
            pred (torch.tensor): predicted values shape (n_samples, n_targets)
            target (torch.tensor): target values shape (n_samples, n_targets)

        Returns:
            torch.tensor: element-wise loss shape (n_samples, n_targets)
        """

        if len(target.shape) == 1:
            target = torch.reshape(target, (len(target), 1))
        return F.mse_loss(pred, target)
