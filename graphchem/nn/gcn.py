r"""MoleculeGCN, graph convolutions on vector representations of molecules"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as gnn


class MoleculeGCN(nn.Module):

    def __init__(self, atom_vocab_size: int, bond_vocab_size: int,
                 output_dim: int, embedding_dim: int = 64, n_messages: int = 2,
                 n_readout: int = 2, readout_dim: int = 64,
                 dropout: float = 0.0):
        """ MoleculeGCN, extends torch.nn.Module; combination of GeneralConv
        and EdgeConv modules and feed-forward readout layer(s) for regressing
        on target variables using molecular structure

        Molecule graphs are first embedded (torch.nn.Embedding), then each
        message passing operation consists of:

        bond_embedding -> EdgeConv -> updated bond_embedding
        atom_embedding + bond_embedding -> GeneralConv -> updated
            atom_embedding

        The sum of all atom states is then passed through a series of fully-
        connected readout layers to regress on a variable:

        atom_embedding -> fully-connected readout layers -> target variable

        Args:
            atom_vocab_size (int): num features (MoleculeEncoder.vocab_sizes)
            bond_vocab_size (int): num features (MoleculeEncoder.vocab_sizes)
            output_dim (int): number of target values per compound
            embedding_dim (int, default=64): number of embedded features for
                atoms and bonds
            n_messages (int, default=2): number of message passes between atoms
            n_readout (int, default=2): number of feed-forward post-readout
                layers (think standard NN/MLP)
            readout_dim (int, default=64): number of neurons in readout layers
            dropout (float, default=0.0): random neuron dropout during training
        """

        super(MoleculeGCN, self).__init__()
        self._dropout = dropout
        self._n_messages = n_messages

        self.emb_atom = nn.Embedding(atom_vocab_size, embedding_dim)
        self.emb_bond = nn.Embedding(bond_vocab_size, embedding_dim)

        self.atom_conv = gnn.GeneralConv(embedding_dim, embedding_dim,
                                         embedding_dim, aggr='add')
        self.bond_conv = gnn.EdgeConv(nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim)
        ))

        self.readout = nn.ModuleList()
        self.readout.append(nn.Sequential(
            nn.Linear(embedding_dim, readout_dim)
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
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        out_atom = self.emb_atom(x)
        out_atom = F.softplus(out_atom)

        out_bond = self.emb_bond(edge_attr)
        out_bond = F.softplus(out_bond)

        for _ in range(self._n_messages):

            out_bond = self.bond_conv(out_bond, edge_index)
            out_bond = F.softplus(out_bond)
            out_bond = F.dropout(out_bond, p=self._dropout,
                                 training=self.training)

            out_atom = self.atom_conv(out_atom, edge_index, out_bond)
            out_atom = F.softplus(out_atom)
            out_atom = F.dropout(out_atom, p=self._dropout,
                                 training=self.training)

        out = gnn.global_add_pool(out_atom, batch)

        for layer in self.readout[:-1]:
            out = layer(out)
            out = F.softplus(out)
            out = F.dropout(out, p=self._dropout, training=self.training)
        out = self.readout[-1](out)

        return (out, out_atom, out_bond)
