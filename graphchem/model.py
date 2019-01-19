from numpy import asarray
from os import environ, mkdir
from os.path import abspath, dirname, isdir
from tensorflow import float32, global_variables_initializer, matmul,\
     placeholder, random_normal, reshape, Session, square, Variable
from tensorflow.nn import relu
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.train import AdamOptimizer, Saver

from graphchem import Graph

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GNN:

    def __init__(self, rnn_size, input_len, output_len, filename='./model'):

        self._weights = Variable(random_normal([rnn_size, output_len]))
        self._biases = Variable(random_normal([output_len]))
        self._input_len = input_len
        self._output_len = output_len
        self._lstm_cell = LSTMCell(
            rnn_size,
            state_is_tuple=True,
            activation=relu
        )
        self._filename = filename
        path = dirname(abspath(self._filename))
        if not isdir(path):
            mkdir(path)

    def single_step(self, x_r):

        x_r = [x_r]
        x = placeholder('float', [1, self._input_len])
        ff = self._feed_forward([x])

        with Session() as sess:
            sess.run(global_variables_initializer())
            o = sess.run([ff], feed_dict={x: x_r})[0][0]
        sess.close()
        if self._output_len == 1:
            return o[0]
        else:
            return o

    def train(self, x_r, y_r, epochs=100):

        x = placeholder(float32, shape=[len(x_r[0]), self._input_len])
        y = placeholder(float32)
        output = self._feed_forward([x])
        cost = square(y - output)
        optimizer = AdamOptimizer().minimize(cost)

        with Session() as sess:
            sess.run(global_variables_initializer())
            for ep in range(epochs):
                epoch_loss = 0
                for idx, a in enumerate(x_r):
                    _, loss = sess.run(
                        [optimizer, cost],
                        feed_dict={x: a, y: y_r[idx]}
                    )
                    epoch_loss += loss
                if ep % 10 == 0:
                    print('Loss: {}'.format(epoch_loss))
            saver = Saver()
            saver.save(sess, self._filename)
        sess.close()

    def use(self, x_r):

        x = placeholder(float32, shape=[len(x_r[0]), self._input_len])
        output = self._feed_forward([x])

        graph_outputs = []
        with Session() as sess:
            saver = Saver()
            saver.restore(sess, self._filename)
            for a in x_r:
                graph_outputs.append(sess.run([output], feed_dict={x: a}))
        sess.close()
        return graph_outputs

    def _feed_forward(self, x):

        outputs, _ = static_rnn(self._lstm_cell, x, dtype=float32)
        return matmul(outputs[-1], self._weights) + self._biases
