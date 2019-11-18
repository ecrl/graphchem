import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

###
from graphchem.molecule import Molecule
###

from csv import DictReader


class MessagePassingNet(nn.Module):

    def __init__(self, atom_state_dim: int, output_dim: int,
                 message_steps: int):

        super(MessagePassingNet, self).__init__()
        self._message_steps = []
        for _ in range(message_steps):
            self._message_steps.append([
                nn.Linear(2 * atom_state_dim, 64),
                nn.Linear(64, 64),
                nn.Linear(64, atom_state_dim)
            ])
        self._fc1 = nn.Linear(atom_state_dim, 64)
        self._fc2 = nn.Linear(64, 64)
        self._out = nn.Linear(64, output_dim)

    def forward(self, molecules: Molecule):
        ### TODO: refactor from single to batches

        out_vals = []
        for mol in molecules:
            new_states = []
            for atom in mol._atoms:
                for ms in self._message_steps:
                    messages = []
                    for con in atom.connectivity:
                        messages.append(F.relu(ms[2](F.relu(ms[1](F.relu(ms[0](
                            torch.tensor(con).float()
                        )))))))
                new_states.append(torch.stack(messages).sum(0))
            for idx, atom in enumerate(mol._atoms):
                atom.state = new_states[idx]
            out_vals.append(F.relu(self._out(F.relu(self._fc2(F.relu(self._fc1(
                torch.stack(new_states)
            )))))).sum(0))
        return torch.stack(out_vals)


class GNN:

    def __init__(self):

        self._model = None

    def train(self, mols, targets, epochs: int=100):

        targets = torch.tensor([[i] for i in targets]).float()
        self._model = MessagePassingNet(
            len(mols[0]._atoms[0].state),
            1, 3
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=0.001)
        for e in range(epochs):
            optimizer.zero_grad()
            output = self._model(mols)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            print('Epoch: {} | Loss: {}'.format(e, loss))

    def predict(self, mols):

        if self._model is None:
            return
        return self._model(mols).tolist()


if __name__ == '__main__':

    with open('cn_data.csv', 'r') as csv_file:
        reader = DictReader(csv_file)
        rows = [r for r in reader]
    csv_file.close()

    smiles = [r['SMILES'] for r in rows]
    ysi = [float(r['CN']) for r in rows]

    mols = [Molecule(smi) for smi in smiles]

    gnn = GNN()
    gnn.train(mols, ysi)
    print(gnn.predict(mols))
