#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# model.py
# TODO: Add support for variable hyperparams (learning rate, layer sizes, etc.)
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#

# Stdlib imports
from os import environ, mkdir
from os.path import abspath, dirname, isdir, isfile

# 3rd party imports
from tensorflow import float32, global_variables_initializer, matmul,\
     placeholder, random_normal, reduce_mean, reduce_sum, reset_default_graph,\
     reshape, Session, square, Variable
from tensorflow.nn import relu
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.train import AdamOptimizer, Saver

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

GLOBAL_HIDDEN_SIZE = 16


class GNN:

    def __init__(self, rnn_size, input_len, num_propagations,
                 filename='./model'):
        '''Graph neural network object

        Args:
            rnn_size (int): size of the LSTM cell's hidden layer
            input_len (int): size of each timestep (length of vector)
            num_propagations (int): number of propagations that occurred
                in the graph
            filename (str): path to save location for the model
        '''

        # Create model path if doesn't exist
        self._filename = filename
        path = dirname(abspath(self._filename))
        if not isdir(path):
            mkdir(path)

        # Reset TensorFlow graph
        reset_default_graph()

        # Shared tensors/operators
        lstm_cell = LSTMCell(
            rnn_size,
            state_is_tuple=True,
            activation=relu
        )
        lstm_weights = Variable(random_normal([rnn_size, int(input_len / 2)]))
        lstm_biases = Variable(random_normal([int(input_len / 2)]))

        # Single-step tensors/operators
        self._x_single = placeholder(float32, [None, input_len])
        rnn_step = static_rnn(
            lstm_cell,
            [self._x_single],
            dtype=float32
        )
        self._output_step = matmul(
            rnn_step[0][-1],
            lstm_weights
        ) + lstm_biases

        # Training tensors/operators
        self._x = placeholder(float32, [num_propagations + 1, input_len])
        self._y = placeholder(float32, 1)
        rnn = static_rnn(
            lstm_cell,
            [self._x],
            dtype=float32
        )
        self._lstm_output = (matmul(
            rnn[0][-1],
            lstm_weights
        ) + lstm_biases)[-1]
        # TODO: Link self._lstm_output to self._g_x (somehow)
        self._g_x = placeholder(float32, [None, int(input_len / 2)])
        g_sum = reduce_sum(self._g_x, 0)
        g_weights_0 = Variable(random_normal(
            [int(input_len / 2), GLOBAL_HIDDEN_SIZE])
        )
        g_biases_0 = Variable(random_normal([GLOBAL_HIDDEN_SIZE]))
        g_weights_1 = Variable(random_normal([GLOBAL_HIDDEN_SIZE, 1]))
        g_hidden = relu(matmul(
            [g_sum], g_weights_0
        ) + g_biases_0)
        self._g_output = matmul(g_hidden, g_weights_1)
        self._cost = reduce_mean(square(self._y - self._g_output))
        self._optimizer = AdamOptimizer().minimize(self._cost)

        # Save the model
        with Session() as sess:
            sess.run(global_variables_initializer())
            saver = Saver()
            if isfile(self._filename + '.meta'):
                saver.restore(sess, self._filename)
            saver.save(sess, self._filename)
        sess.close()

    def single_step(self, x_r):
        '''Performs a "single step" through the LSTM cell, returning last
        LSTM outputs

        Args:
            x_r (iterable): single-state vector

        Returns:
            list: output of x_r -> LSTM cell
        '''

        with Session() as sess:
            saver = Saver()
            saver.restore(sess, self._filename)
            output = sess.run(
                [self._output_step],
                feed_dict={self._x_single: x_r}
            )[0]
        sess.close()
        return output

    def train(self, graphs, targets, epochs=100):
        '''Trains model using supplied graphs and targets (NOT COMPLETE)

        Args:
            graphs (list): list of graphchem.Graph.feeds, each feed containing
                all atom states after propagation -> single-step
            targets (list): list of target values, equal in length to graphs
            epochs (int): number of training iterations
        '''

        with Session() as sess:
            saver = Saver()
            saver.restore(sess, self._filename)
            for e in range(epochs):
                for idx, graph in enumerate(graphs):
                    graph_repr = []
                    for a in graph:
                        graph_repr.append(sess.run(
                            self._lstm_output, feed_dict={
                                self._x: a
                            }
                        ))
                    _, loss = sess.run(
                        [self._optimizer, self._cost],
                        feed_dict={
                            self._g_x: graph_repr,
                            self._y: [targets[idx]]
                        }
                    )
                    # TODO: LSTM model (occurs at every node) doesn't train
                    #   here, only the global output function (occurs on
                    #   reduced sum of node LSTM outputs). I need to make LSTM
                    #   training work.
                if e % 10 == 0:
                    print('Loss: {}'.format(loss))
            saver.save(sess, self._filename)
        sess.close()

    def use(self, graph):
        '''Predicts value for supplied graph

        Args:
            graph: a chemgraph.Graph.feeds list, containing all atom states
                after propagation -> single-step

        Returns:
            float: predicted value
        '''

        with Session() as sess:
            saver = Saver()
            saver.restore(sess, self._filename)
            graph_repr = []
            for a in graph:
                graph_repr.append(sess.run(self._lstm_output, feed_dict={
                    self._x: a
                }))
            output = sess.run(self._g_output, feed_dict={
                self._g_x: graph_repr
            })
        sess.close()
        return output[0][0]