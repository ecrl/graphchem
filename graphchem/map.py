from graphchem import Graph
from graphchem import GNN


class GraphMapper:

    def __init__(self, rnn_size=32):

        self._graphs = []
        self._targets = []
        self._nn = None
        self._rnn_size = rnn_size

    def add_graph(self, smiles, target_val):

        self._graphs.append(Graph(smiles))
        self._targets.append(target_val)

    def remove_all_graphs(self):

        self._graphs = []
        self._targets = []

    def train(self, num_propagations=2, epochs=100):

        if len(self._graphs) == 0:
            raise RuntimeError('No graphs to train')

        self._nn = GNN(self._rnn_size, len(self._graphs[0].current_repr[0]), 1)

        for g in self._graphs:
            for _ in range(num_propagations + 1):
                g.propagate(self._nn.single_step)

        inputs = []
        outputs = []
        for idx, graph in enumerate(self._graphs):
            for _ in range(len(graph)):
                outputs.extend([self._targets[idx]])
            inputs.extend(graph.model_repr)

        self._nn.train(inputs, outputs, epochs=epochs)

    def predict(self):

        inputs = []
        for graph in self._graphs:
            inputs.extend(graph.model_repr)
        return self._nn.use(inputs)


if __name__ == '__main__':

    gm = GraphMapper()
    gm.add_graph('COCOCOC', 7)
    gm.train(epochs=500)
    print(gm.predict())
