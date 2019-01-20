from graphchem import GraphMapper

if __name__ == '__main__':

    gm = GraphMapper(num_propagations=2, rnn_size=48)
    gm.add_graph('COCOCOC', 7)
    gm.train(epochs=500)
    preds = gm.predict()
    for p in preds:
        print(p)
