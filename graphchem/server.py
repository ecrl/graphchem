from torch import device, cuda
from torch.nn.functional import mse_loss
from torch.optim import Adam

from graphchem.chem.molecule import Molecule
from graphchem.models.gnn import MessagePassingNet
from graphchem.models.preprocessing import dataloader_from_mols


class Server:

    def __init__(self):

        self._model = None

    def train(self, smiles, targets, epochs: int = 250, lr: float = 0.01,
              batch_size: int = None):

        mol_graphs = [Molecule(smi) for smi in smiles]

        if type(targets[0]) != list:
            for idx in range(len(targets)):
                targets[idx] = [targets[idx]]

        if batch_size is None:
            batch_size = len(mol_graphs)
        data = dataloader_from_mols(mol_graphs, targets, batch_size=batch_size)

        dev = device('cuda' if cuda.is_available() else 'cpu')
        self._model = MessagePassingNet(len(mol_graphs[0]._atoms[0].state),
                                        len(targets[0])).to(dev)
        opt = Adam(self._model.parameters(), lr=lr)

        for e in range(epochs):
            self._model.train()
            loss_all = 0
            for d in data:
                d = d.to(dev)
                opt.zero_grad()
                output = self._model(d)
                loss = mse_loss(output, d.y.float())
                loss.backward()
                loss_all += d.num_graphs * loss.item()
                opt.step()
            loss = loss_all / len(data)
            print('Epoch: {} Loss: {}'.format(e, loss))

    def predict(self, smiles):

        if self._model is None:
            raise RuntimeError('Model is not trained yet')

        mol_graphs = [Molecule(smi) for smi in smiles]
        data = dataloader_from_mols(mol_graphs, batch_size=len(mol_graphs))
        return [self._model(d).tolist() for d in data]
