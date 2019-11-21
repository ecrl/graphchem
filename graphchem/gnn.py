#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# gnn.py
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#

# 3rd party imports
from torch import stack, tensor
from torch.nn import Linear, Module, MSELoss
from torch.nn.functional import relu
from torch.optim import Adam

# custom imports
from graphchem.molecule import Molecule


class MessagePassingNet(Module):

    def __init__(self, atom_state_dim: int, output_dim: int,
                 message_steps: int, message_n_hidden: int,
                 message_size_hidden: int, readout_n_hidden: int,
                 readout_size_hidden: int):
        ''' MessagePassingNet: extends torch.nn.Module: implementation of
        neural message passing, including atom-centric message passing,
        molecule readout

        Args:
            atom_state_dim (int): number of numerical features per atom (i.e.
                len(graphchem.Atom.state))
            output_dim (int): dimensionality of target values
            message_steps (int): number of independent message-passing steps
            message_n_hidden (int): number of hidden layers/message pass
            message_size_hidden (int): number of neurons per message hidden
            readout_n_hidden (int): number of hidden layers in molecule readout
            readout_size_hidden (int): number of neurons per readout hidden
        '''

        super(MessagePassingNet, self).__init__()

        self._message_steps = message_steps
        self._message_n_hidden = message_n_hidden
        self._readout_n_hidden = readout_n_hidden
        for step in range(message_steps):
            self.__setattr__(
                '_ms{}_in'.format(step),
                Linear(2 * atom_state_dim, message_size_hidden)
            )
            for n_hidden in range(message_n_hidden):
                self.__setattr__(
                    '_ms{}_h{}'.format(step, n_hidden),
                    Linear(message_size_hidden, message_size_hidden)
                )
            self.__setattr__(
                '_ms{}_out'.format(step),
                Linear(message_size_hidden, atom_state_dim)
            )
        self._ro_in = Linear(atom_state_dim, readout_size_hidden)
        for n_hidden in range(readout_n_hidden):
            self.__setattr__(
                '_ro_h{}'.format(n_hidden),
                Linear(readout_size_hidden, readout_size_hidden)
            )
        self._ro_out = Linear(readout_size_hidden, output_dim)

    def forward(self, molecules: list) -> tensor:
        ''' MessagePassingNet.forward: feed forward implementation, as follows:

        1.) perform message passing for each atom: messages contain the atom's
            state and its neighbors' states (2 x atom state length, one
            message per neighbor). Messages are passed through densely-
            connected layer(s), resulting in dimensionality of size
            len(atom.state).

        2.) aggregation of passed messages: new atom state = sum(messages
            passed through dense layers).

        3.) repeat steps 1 and 2 for each message passing step; note, each
            densely connected series is independent.

        4.) perform readout: molecule representation is defined as the sum of
            all atom states; this sum as passed through densely connected
            layers until output dimensionality is achieved.

        Args:
            molecules (list): list of graphchem.Molecule objects

        Returns:
            torch.tensor: result of feeding forward supplied molecules
        '''

        out_vals = []
        for mol in molecules:
            for step in range(self._message_steps):
                new_states = []
                for atom in mol._atoms:
                    messages = []
                    for con in atom.connectivity:
                        m = tensor(con).float()
                        m = relu(getattr(self, '_ms{}_in'.format(step))(m))
                        for h in range(self._message_n_hidden):
                            m = relu(getattr(
                                self, '_ms{}_h{}'.format(step, h)
                            )(m))
                        m = relu(getattr(self, '_ms{}_out'.format(step))(m))
                        messages.append(m)
                    new_states.append(stack(messages).sum(0))
                for idx, atom in enumerate(mol._atoms):
                    atom.state = new_states[idx]
            final_atom_states = stack([a.state for a in mol._atoms])
            mol_repr = final_atom_states.sum(0)
            out = relu(self._ro_in(mol_repr))
            for h in range(self._readout_n_hidden):
                out = relu(getattr(self, '_ro_h{}'.format(h))(out))
            out_vals.append(self._ro_out(out))
        return stack(out_vals)


class GNN:

    def __init__(self, config: dict = None):
        ''' GNN: implementation of graph neural network for operation on
        molecular data. Utilizes MessagePassingNet for training, predictions.

        Args:
            config (dict): if None, default config (GNN._default_config())
                is used. Else, sets config to supplied values.
        '''

        self._model = None
        if config is None:
            config = self._default_config()
        self._conf = config

    def train(self, mols: list, targets: list, epochs: int = 300,
              verbose: bool = False):
        ''' GNN.train: trains GNN on supplied molecules, target data.

        Args:
            mols (list): list of graphchem.Molecule objects
            targets (list): list of target values, int/float; can be any
                dimension (i.e. list elements are sublists)
            epochs (int): number of training iterations
            verbose (bool): if true, displays loss after each epoch
        '''

        if type(targets[0]) is not list:
            n_output = 1
            targets = tensor([[i] for i in targets]).float()
        else:
            targets = tensor([i for i in targets]).float()
            n_output = targets.shape[-1]
        self._model = MessagePassingNet(
            len(mols[0]._atoms[0].state),
            n_output,
            self._conf['message_steps'],
            self._conf['message_n_hidden'],
            self._conf['message_size_hidden'],
            self._conf['readout_n_hidden'],
            self._conf['readout_size_hidden']
        )
        criterion = MSELoss()
        optimizer = Adam(
            self._model.parameters(),
            lr=self._conf['learning_rate'],
            weight_decay=self._conf['weight_decay']
        )
        for e in range(epochs):
            optimizer.zero_grad()
            output = self._model(mols)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            print('Epoch: {} | Loss: {}'.format(e, loss))

    def predict(self, mols: list) -> list:
        ''' GNN.predict: predicts values for supplied molecules using trained
        model

        Args:
            mols (list): list of graphchem.Molecule objects

        Returns:
            list: predicted values for supplied molecules
        '''

        if self._model is None:
            raise RuntimeError('GNN has not been trained (GNN.train)')
        return self._model(mols).tolist()

    @staticmethod
    def _default_config() -> dict:
        ''' GNN._default_config: default configuration for MessagePassingNet
        architecture, learning parameters

        Returns:
            dict: default configuration variables
        '''

        return {
            'message_steps': 3,
            'message_n_hidden': 1,
            'message_size_hidden': 64,
            'readout_n_hidden': 1,
            'readout_size_hidden': 64,
            'learning_rate': 0.001,
            'weight_decay': 0.0
        }
