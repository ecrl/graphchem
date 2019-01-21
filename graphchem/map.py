#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# map.py
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#

# GraphChem imports
from graphchem import Graph
from graphchem import GNN


class GraphMapper:

    def __init__(self, num_propagations=2, rnn_size=32,
                 model_filename='./model/model'):
        '''GraphMapper object: maps graph(s) to target values

        Args:
            num_propagations (int): number of times each graph's nodes are
                allowed to propagate their information to neighbor nodes
            rnn_size (int): size of LSTM cell's hidden layer
            model_filename (str): path to model save location
        '''

        self._graphs = []
        self._targets = []
        self._nn = None
        self._num_propagations = num_propagations
        self._rnn_size = rnn_size
        self._model_filename = model_filename

    def add_graph(self, smiles, target_val):
        '''Adds a graph to the GraphMapper

        Args:
            smiles (str): molecule SMILES string
            target_val (num): target value (e.g. number of atoms)
        '''

        self._graphs.append(Graph(smiles))
        self._targets.append(target_val)

    def remove_all_graphs(self):
        '''Removes all graphs/target values from the GraphMapper'''

        self._graphs = []
        self._targets = []

    def train(self, epochs=100):
        '''Trains the GraphMapper's graph neural network model using currently
        supplied graphs

        Args:
            epochs (int): number of training iterations
        '''

        if self._nn is None:
            self._init_model()
        self._reset_graphs()
        self._nn.train(
            [g.feeds for g in self._graphs],
            self._targets,
            epochs=epochs
        )

    def predict(self):
        '''Predicts values for all currently supplied graphs

        Returns:
            list: list of predictions corresponding to supplied graphs
        '''

        if self._nn is None:
            self._init_model()
        self._reset_graphs()
        predictions = []
        for graph in self._graphs:
            predictions.append(self._nn.use(graph.feeds))
        return predictions

    def _reset_graphs(self):
        '''Reset the states and feeds for all currently supplied graphs'''

        for graph in self._graphs:
            graph.reset_graph()
            for _ in range(self._num_propagations + 1):
                # TODO: multiprocess this
                graph.propagate(self._nn.single_step)

    def _init_model(self):
        '''Called if the graph neural network has not been initialized'''

        self._nn = GNN(
            self._rnn_size,
            self._graphs[0].feed_len,
            self._num_propagations,
            filename=self._model_filename
        )


if __name__ == '__main__':

    gm = GraphMapper()
    gm.add_graph('COCOCOC', 7)
    gm.add_graph('CCOCC', 5)
    gm.train(epochs=300)
    preds = gm.predict()
    print(preds)
