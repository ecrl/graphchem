from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GeneralConv, EdgeConv, global_add_pool


class MoleculeGCN(nn.Module):
    """
    A Graph Convolutional Network (GCN) model for molecular property
    prediction.

    Attributes
    ----------
    _p_dropout : float
        Probability of an element to be zeroed in dropout layers.
    _n_messages : int
        Number of message passing steps.
    emb_atom : nn.Embedding
        Embedding layer for atoms.
    emb_bond : nn.Embedding
        Embedding layer for bonds.
    atom_conv : GeneralConv
        General convolution layer for atoms.
    bond_conv : EdgeConv
        Edge convolution layer for bonds.
    readout : nn.ModuleList
        Readout network consisting of fully connected layers.
    """

    def __init__(self, atom_vocab_size: int, bond_vocab_size: int,
                 output_dim: int, embedding_dim: Optional[int] = 128,
                 n_messages: Optional[int] = 2,
                 n_readout: Optional[int] = 2,
                 readout_dim: Optional[int] = 64,
                 p_dropout: Optional[float] = 0.0,
                 aggr: Optional[str] = "add"):
        """
        Initialize the MoleculeGCN object.

        Parameters
        ----------
        atom_vocab_size : int
            Number of unique atom representations in the dataset.
        bond_vocab_size : int
            Number of unique bond representations in the dataset.
        output_dim : int
            Dimensionality of the output space.
        embedding_dim : int, optional (default=128)
            Dimensionality of the atom and bond embeddings.
        n_messages : int, optional (default=2)
            Number of message passing steps.
        n_readout : int, optional (default=2)
            Number of fully connected layers in the readout network.
        readout_dim : int, optional (default=64)
            Dimensionality of the hidden layers in the readout network.
        p_dropout : float, optional (default=0.0)
            Dropout probability for the dropout layers.
        aggr : str, optional (default="add")
            Aggregation scheme to use in the GeneralConv layer.
        """
        super().__init__()

        self._p_dropout = p_dropout
        self._n_messages = n_messages

        self.emb_atom = nn.Embedding(atom_vocab_size, embedding_dim)
        self.emb_bond = nn.Embedding(bond_vocab_size, embedding_dim)

        self.atom_conv = GeneralConv(
            embedding_dim, embedding_dim, embedding_dim, aggr=aggr
        )
        self.bond_conv = EdgeConv(nn.Sequential(
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

    def forward(
            self,
            data: torch_geometric.data.Data
         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MoleculeGCN.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data containing node features (x), edge attributes
            (edge_attr), edge indices (edge_index), and batch vector (batch).

        Returns
        -------
        out : torch.Tensor
            The final output predictions for the input molecules.
        out_atom : torch.Tensor
            Atom-level representations after message passing.
        out_bond : torch.Tensor
            Bond-level representations after message passing.
        """
        x, edge_attr, edge_index, batch = data.x, data.edge_attr, \
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
            out_bond = F.dropout(
                out_bond, p=self._p_dropout, training=self.training
            )

            out_atom = self.atom_conv(out_atom, edge_index, out_bond)
            out_atom = F.softplus(out_atom)
            out_atom = F.dropout(
                out_atom, p=self._p_dropout, training=self.training
            )

        out = global_add_pool(out_atom, batch)

        for layer in self.readout[:-1]:

            out = layer(out)
            out = F.softplus(out)
            out = F.dropout(
                out, p=self._p_dropout, training=self.training
            )

        out = self.readout[-1](out)

        return (out, out_atom, out_bond)
