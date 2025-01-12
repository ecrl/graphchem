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
    act_fn : callable
        Activation function, e.g., `torch.nn.functional.softplus`
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
                 aggr: Optional[str] = "add",
                 act_fn: Optional[callable] = F.softplus):
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
        act_fn : callable, optional (default=`torch.nn.functional.softplus`)
            Activation function, e.g., `torch.nn.functional.softplus`,
            `torch.nn.functional.sigmoid`, `torch.nn.functional.relu`, etc.
        """
        super().__init__()

        # Store attributes
        self._p_dropout = p_dropout
        self._n_messages = n_messages
        self.act_fn = act_fn

        # Embedding layer for atoms
        self.emb_atom = nn.Embedding(atom_vocab_size, embedding_dim)

        # Embedding layer for bonds
        self.emb_bond = nn.Embedding(bond_vocab_size, embedding_dim)

        # General convolution layer for atoms with specified aggregation method
        self.atom_conv = GeneralConv(
            embedding_dim, embedding_dim, embedding_dim, aggr=aggr
        )

        # Edge convolution layer for bonds using a linear transformation
        self.bond_conv = EdgeConv(nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim)
        ))

        # Initialize the readout network if readout layers are specified
        if n_readout > 0:

            # Create a list to hold the readout network modules
            self.readout = nn.ModuleList()

            # First layer of the readout network
            self.readout.append(nn.Sequential(
                nn.Linear(embedding_dim, readout_dim)
            ))

            # Additional hidden layers for the readout network if needed
            if n_readout > 1:
                for _ in range(n_readout - 1):
                    self.readout.append(nn.Sequential(
                        nn.Linear(readout_dim, readout_dim)
                    ))

            # Final layer of the readout network to produce output dimensions
            self.readout.append(nn.Sequential(
                nn.Linear(readout_dim, output_dim)
            ))

        # No readout network if n_readout is 0
        else:
            self.readout = None

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
        # Extract node features, edge attributes, edge indices, and batch
        #   vector from data
        x, edge_attr, edge_index, batch = data.x, data.edge_attr, \
            data.edge_index, data.batch

        # If no node features are provided, initialize with ones
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        # Embed and activate atom features
        out_atom = self.emb_atom(x)
        out_atom = self.act_fn(out_atom)

        # Embed and activate bond features
        out_bond = self.emb_bond(edge_attr)
        out_bond = self.act_fn(out_bond)

        # Perform message passing for the specified number of steps
        for _ in range(self._n_messages):

            # Update bond representations using edge convolution
            out_bond = self.bond_conv(out_bond, edge_index)
            out_bond = self.act_fn(out_bond)

            # Apply dropout
            out_bond = F.dropout(
                out_bond, p=self._p_dropout, training=self.training
            )

            # Update atom representations using general convolution
            out_atom = self.atom_conv(out_atom, edge_index, out_bond)
            out_atom = self.act_fn(out_atom)

            # Apply dropout
            out_atom = F.dropout(
                out_atom, p=self._p_dropout, training=self.training
            )

        # Aggregate atom representations across batches with global add pooling
        out = global_add_pool(out_atom, batch)

        # Process aggregated atom representation through the readout network
        if self.readout is not None:

            # Iterate over all but the last layer of the readout network
            for layer in self.readout[:-1]:

                # Pass through layer and activate
                out = layer(out)
                out = self.act_fn(out)

                # Apply dropout
                out = F.dropout(
                    out, p=self._p_dropout, training=self.training
                )

            # Final layer of the readout network to produce output dimensions
            out = self.readout[-1](out)

        # Return final prediction, atom representations, bond representations
        return (out, out_atom, out_bond)
