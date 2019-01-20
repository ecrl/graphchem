from numpy import asarray
from os import environ, mkdir
from os.path import abspath, dirname, isdir, isfile
from tensorflow import float32, global_variables_initializer, matmul,\
     placeholder, random_normal, reset_default_graph, reshape, Session,\
     square, Variable
from tensorflow.nn import relu
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.train import AdamOptimizer, Saver

from graphchem import Graph

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GNN:

    def __init__(self, rnn_size, input_len, output_len, num_propagations,
                 filename='./model'):

        self._filename = filename
        path = dirname(abspath(self._filename))
        if not isdir(path):
            mkdir(path)

        reset_default_graph()
        self._x_single = placeholder(float32, [1, input_len])
        self._x = placeholder(float32, [num_propagations + 1, input_len])
        self._y = placeholder(float32)
        self._lstm_cell = LSTMCell(
            rnn_size,
            state_is_tuple=True,
            activation=relu
        )
        self._rnn_step = static_rnn(
            self._lstm_cell,
            [self._x_single],
            dtype=float32
        )
        self._rnn = static_rnn(
            self._lstm_cell,
            [self._x],
            dtype=float32
        )
        self._weights = Variable(random_normal([rnn_size, output_len]))
        self._biases = Variable(random_normal([output_len]))
        self._output_step = matmul(
            self._rnn_step[0][-1],
            self._weights
        ) + self._biases
        self._output = matmul(
            self._rnn[0][-1],
            self._weights
        ) + self._biases
        self._cost = square(self._y - self._output)
        self._optimizer = AdamOptimizer().minimize(self._cost)
        with Session() as sess:
            sess.run(global_variables_initializer())
            saver = Saver()
            if isfile(self._filename + '.meta'):
                saver.restore(sess, self._filename)
            saver.save(sess, self._filename)
        sess.close()

    def single_step(self, x_r):

        with Session() as sess:
            saver = Saver()
            saver.restore(sess, self._filename)
            o = sess.run(
                [self._output_step],
                feed_dict={self._x_single: x_r}
            )[0][0]
        sess.close()
        return o

    def train(self, x_r, y_r, epochs=100):

        with Session() as sess:
            saver = Saver()
            saver.restore(sess, self._filename)
            for ep in range(epochs):
                epoch_loss = 0
                # TODO: batch training
                for idx, a in enumerate(x_r):
                    _, loss = sess.run(
                        [self._optimizer, self._cost],
                        feed_dict={self._x: a, self._y: y_r[idx]}
                    )
                    epoch_loss += loss
                if ep % 10 == 0:
                    print('Loss: {}'.format(epoch_loss))
            saver = Saver()
            saver.save(sess, self._filename)
        sess.close()

    def use(self, x_r):

        graph_outputs = []
        with Session() as sess:
            saver = Saver()
            saver.restore(sess, self._filename)
            for a in x_r:
                graph_outputs.append(sess.run(
                    [self._output],
                    feed_dict={self._x: a}
                )[0][-1][0])
        sess.close()
        return graph_outputs
