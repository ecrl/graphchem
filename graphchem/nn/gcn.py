from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as gnn
from torch_scatter import scatter_add


class MoleculeGCN(nn.Module):

    def __init__(self, atom_dim: int, bond_dim: int, output_dim: int,
                 embedding_dim: int = 64, n_messages: int = 2,
                 n_readout: int = 2, readout_dim: int = 64,
                 dropout: float = 0.0):
        """ MoleculeGCN, extends torch.nn.Module; combination of MFConv and
        EdgeConv modules, two GRUs, scatter add, and feed-forward readout
        layer(s) such that:

        atom_features -> embedding -> MFConv -> atom GRU -> readout
        bond_features -> embedding -> EdgeConv -> bond GRU -> readout

        readout -> scatter add over atoms/nodes -> feed-forward layer(s)
        -> target(s)

        Args:
            atom_dim (int): number of features per atom
            bond_dim (int): number of features per bond
            output_dim (int): number of target values per compound
            embedding_dim (int, default=16): number of embedded features for
                atoms and bonds
            n_messages (int, default=2): number of message passes between atoms
            n_readout (int, default=1): number of feed-forward post-readout
                layers (think standard NN/MLP)
            readout_dim (int, default=8): number of neurons in readout layers
            dropout (float, default=0.0): random neuron dropout during training
        """

        super(MoleculeGCN, self).__init__()
        self._dropout = dropout
        self._n_messages = n_messages

        self.emb_atom = nn.Linear(atom_dim, embedding_dim)
        self.emb_bond = nn.Linear(bond_dim, embedding_dim)

        self.atom_conv = gnn.MFConv(embedding_dim, embedding_dim)
        self.bond_conv = gnn.EdgeConv(nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim)
        ))
        self.atom_gru = nn.GRU(embedding_dim, embedding_dim)
        self.bond_gru = nn.GRU(embedding_dim, embedding_dim)

        self.readout = nn.ModuleList()
        self.readout.append(nn.Sequential(
            nn.Linear(2 * embedding_dim, readout_dim)
        ))
        if n_readout > 1:
            for _ in range(n_readout - 1):
                self.readout.append(nn.Sequential(
                    nn.Linear(readout_dim, readout_dim)
                ))
        self.readout.append(nn.Sequential(
            nn.Linear(readout_dim, output_dim)
        ))

    def forward(self, data: 'torch_geometric.data.Data') -> Tuple[
     'torch.tensor', 'torch.tensor', 'torch.tensor']:
        """ forward operation for PyTorch module; given a sample of
        torch_geometric.data.Data, with atom/bond attributes and connectivity,
        perform message passing operations and readout

        Args:
            data (torch_geometric.data.Data): torch_geometric data object or
                inheritee

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor]: (readout output
            (target prediction), atom embeddings, bond embeddings); embeddings
            represent pre-sum/readout values present at each atom/bond, useful
            for determining which atoms/bonds contribute to target value
        """

        x, edge_attr, edge_index, batch = data.x, data.edge_attr,\
            data.edge_index, data.batch
        row, col = edge_index
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        out = F.relu(self.emb_atom(x))
        out_edge = F.relu(self.emb_bond(edge_attr))
        h = out.unsqueeze(0)
        h_edge = out_edge.unsqueeze(0)

        for _ in range(self._n_messages):

            m = F.relu(self.atom_conv(out, edge_index))
            emb_node = m
            m = F.dropout(m, p=self._dropout, training=self.training)
            out, h = self.atom_gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

            m_edge = F.relu(self.bond_conv(out_edge, edge_index))
            emb_edge = m_edge
            m_edge = F.dropout(m_edge, p=self._dropout, training=self.training)
            out_edge, h_edge = self.bond_gru(m_edge.unsqueeze(0), h_edge)
            out_edge = out_edge.squeeze(0)

        out = torch.cat([out[row], out_edge[col]], dim=1)
        out = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out = gnn.global_add_pool(out, batch)

        for layer in self.readout[:-1]:
            out = layer(out)
            out = F.relu(out)
            out = F.dropout(out, p=self._dropout, training=self.training)
        out = self.readout[-1](out)

        return (out, emb_node, emb_edge)
