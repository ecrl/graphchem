import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from graphchem.molecule import Molecule

from csv import DictReader


class MessagePassingNet(nn.Module):

    def __init__(self, atom_state_dim: int, output_dim: int,
                 message_steps: int):

        super(MessagePassingNet, self).__init__()

        self._message_steps = message_steps
        for step in range(message_steps):
            self.__setattr__(
                '_ms{}_in'.format(step),
                nn.Linear(2 * atom_state_dim, 64)
            )
            self.__setattr__(
                '_ms{}_h'.format(step),
                nn.Linear(64, 64)
            )
            self.__setattr__(
                '_ms{}_out'.format(step),
                nn.Linear(64, atom_state_dim)
            )

        self._fc1 = nn.Linear(atom_state_dim, 64)
        self._fc2 = nn.Linear(64, 64)
        self._out = nn.Linear(64, output_dim)

    def forward(self, molecules: Molecule):

        out_vals = []
        for mol in molecules:
            for step in range(self._message_steps):
                new_states = []
                for atom in mol._atoms:
                    messages = []
                    for con in atom.connectivity:
                        m = torch.tensor(con).float()
                        m = F.relu(getattr(self, '_ms{}_in'.format(step))(m))
                        m = F.relu(getattr(self, '_ms{}_h'.format(step))(m))
                        m = F.relu(getattr(self, '_ms{}_out'.format(step))(m))
                        messages.append(m)
                    new_states.append(torch.stack(messages).sum(0))
                for idx, atom in enumerate(mol._atoms):
                    atom.state = new_states[idx]
            final_atom_states = torch.stack([a.state for a in mol._atoms])
            mol_repr = final_atom_states.sum(0)
            out_vals.append(self._out(F.relu(self._fc2(F.relu(self._fc1(
                mol_repr
            ))))))
        return torch.stack(out_vals)


class GNN:

    def __init__(self):

        self._model = None

    def train(self, mols, targets, epochs: int = 300):

        if type(targets[0]) is not list:
            n_output = 1
            targets = torch.tensor([[i] for i in targets]).float()
        else:
            targets = torch.tensor([i for i in targets]).float()
            n_output = targets.shape[-1]
        self._model = MessagePassingNet(
            len(mols[0]._atoms[0].state),
            n_output, 3
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=0.01)
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

    smiles = [
        'CCC',
        'CCCCC',
        'CCCCCCC'
    ]
    mols = [Molecule(smi) for smi in smiles]

    targets = [3, 5, 7]

    gnn = GNN()
    gnn.train(mols, targets)

    new_smiles = [
        'CCCC',
        'CCCCCC'
    ]
    new_mols = [Molecule(smi) for smi in new_smiles]
    print(gnn.predict(new_mols))
