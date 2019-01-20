from graphchem import Graph
from graphchem import GNN


class GraphMapper:

    def __init__(self, num_propagations=2, rnn_size=32):

        self._graphs = []
        self._targets = []
        self._nn = None
        self._num_propagations = num_propagations
        self._rnn_size = rnn_size

    def add_graph(self, smiles, target_val):

        self._graphs.append(Graph(smiles))
        self._targets.append([target_val])

    def remove_all_graphs(self):

        self._graphs = []
        self._targets = []

    def train(self, epochs=100):

        if len(self._graphs) == 0:
            raise RuntimeError('No graphs to train')

        if self._nn is None:
            self._init_model()

        self._reset_graphs()
        inputs = []
        outputs = []
        for idx, graph in enumerate(self._graphs):
            for _ in range(len(graph)):
                outputs.extend(self._targets[idx])
            inputs.extend(graph.model_repr)

        self._nn.train(inputs, outputs, epochs=epochs)

    def predict(self):

        if len(self._graphs) == 0:
            raise RuntimeError('No graphs to train')

        if self._nn is None:
            self._init_model()

        self._reset_graphs()
        inputs = []
        for graph in self._graphs:
            inputs.extend(graph.model_repr)
        return self._nn.use(inputs)

    def _reset_graphs(self):

        for graph in self._graphs:
            graph.reset_states()
            for _ in range(self._num_propagations + 1):
                graph.propagate(self._nn.single_step)

    def _init_model(self):

        self._nn = GNN(
            self._rnn_size,
            len(self._graphs[0].current_repr[0]),
            1,
            self._num_propagations
        )
