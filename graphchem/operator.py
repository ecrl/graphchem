import torch
import torch_geometric.data as gdata
from sklearn.model_selection import train_test_split

import warnings

from .preprocessing import CompoundEncoder
from .models import MessagePassingNet, LRDecayLinear, CallbackOperator,\
    Validator


def _default_config():

    return {
        'task': 'graph',
        'valid_size': 0.2,
        'valid_epoch_iter': 4,
        'valid_patience': 8,
        'batch_size': 1,
        'learning_rate': 0.001,
        'lr_decay': 0.0,
        'epochs': 32,
        'verbose': 0,
        'device': torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
    }


class GraphOperator(object):

    def __init__(self, config: dict = None):

        if config is None:
            config = _default_config()
        else:
            df = _default_config()
            for key in df.keys():
                try:
                    _val = config[key]
                except KeyError:
                    config[key] = df[key]
                    warnings.warn(
                        '{} config value not found: default value set, {}'
                        .format(key, df[key])
                    )
        self._config = config
        self._model = None
        self._ce = None

    def train(self, smiles: list, target: list, model_filename: str = None,
              model_config: dict = None):
        ''' GraphOperator.train: trains a graph neural network given SMILES
        strings, target values, supplied config (i.e. architecture, hyper-
        parameters)

        Args:
            smiles (list): list of SMILES strings (str)
            target (list): list of target values (1d, float)
            model_filename (str): if not None, saves model to this location
            model_config (dict): configuration dict; if none supplied, default
                is used

        Returns:
            None
        '''

        # Check for inequality in length of input, target data
        if len(smiles) != len(target):
            raise ValueError(
                'Supplied SMILES and targets not the same length: {}, {}'
                .format(len(smiles), len(target))
            )

        # Prepare data
        self._ce = CompoundEncoder(smiles)
        data = []
        for idx, smi in enumerate(smiles):
            a, b = self._ce.encode(smi)
            data.append(gdata.Data(
                x=a,
                edge_index=self._ce.connectivity(smi),
                edge_attr=b,
                y=torch.tensor(target[idx]).type(torch.float)
            ).to(self._config['device']))

        # Split data into training, validation subsets
        data_train, data_valid = train_test_split(
            data, test_size=self._config['valid_size']
        )
        loader_train = gdata.DataLoader(
            data_train,
            batch_size=self._config['batch_size'],
            shuffle=True
        )
        loader_valid = gdata.DataLoader(
            data_valid,
            batch_size=self._config['batch_size'],
            shuffle=True
        )

        # Create model
        self._model = MessagePassingNet(
            self._ce.ATOM_DIM,
            len(target[0]),
            task=self._config['task'],
            config=model_config
        )
        self._model.construct()
        self._model.to(self._config['device'])
        optimizer = torch.optim.Adam(self._model.parameters(),
                                     lr=self._config['learning_rate'])

        # Setup callbacks
        CBO = CallbackOperator()
        _lrdecay = LRDecayLinear(
            self._config['learning_rate'],
            self._config['lr_decay'],
            optimizer
        )
        _validator = Validator(
            loader_valid,
            self._model,
            self._config['valid_epoch_iter'],
            self._config['valid_patience']
        )
        CBO.add_cb(_lrdecay)
        CBO.add_cb(_validator)

        # TRAIN BEGIN
        CBO.on_train_begin()

        # Begin training loop
        for epoch in range(self._config['epochs']):

            # EPOCH BEGIN
            if not CBO.on_epoch_begin(epoch):
                break

            train_loss = 0.0
            self._model.train()

            for b_idx, batch in enumerate(loader_train):

                # BATCH BEGIN
                if not CBO.on_batch_begin(b_idx):
                    break

                optimizer.zero_grad()
                embedding, pred = self._model(batch)
                target = batch.y
                if self._config['task'] == 'node':
                    pred = pred[batch.train_mask]
                    target = target[batch.train_mask]

                # BATCH END, LOSS BEGIN
                if not CBO.on_batch_end(b_idx):
                    break
                if not CBO.on_loss_begin(b_idx):
                    break

                loss = self._model.loss(pred, target)
                loss.backward()

                # LOSS END, STEP BEGIN
                if not CBO.on_loss_end(b_idx):
                    break
                if not CBO.on_step_begin(b_idx):
                    break

                optimizer.step()
                train_loss += loss.detach().item() * batch.num_graphs

                # STEP END
                if not CBO.on_step_end(b_idx):
                    break

            train_loss /= len(loader_train.dataset)

            # EPOCH END
            if not CBO.on_epoch_end(epoch):
                break

            if self._config['verbose']:
                print('Epoch: {} | Train Loss: {} | Valid Loss: {}'.format(
                      epoch, train_loss, _validator._best_loss))

        # TRAIN END
        CBO.on_train_end()

        if model_filename is not None:
            torch.save(self._model, model_filename)

    def use(self, smiles: list, model_filename=None) -> list:

        # Figure out what to use
        if self._model is None and model_filename is None:
            raise RuntimeError(
                'Model not previously built, or model not supplied'
            )
        if model_filename is not None:
            self._model = torch.load(model_filename)
            self._model.eval()

        # Prepare data
        data = []
        for idx, smi in enumerate(smiles):
            a, b = self._ce.encode(smi)
            data.append(gdata.Data(
                x=a,
                edge_index=self._ce.connectivity(smi),
                edge_attr=b
            ).to(self._config['device']))
        loader_test = gdata.DataLoader(
            data,
            batch_size=1,
            shuffle=False
        )

        # Get results
        results = []
        for batch in loader_test:
            _, res = self._model(batch)
            results.append(res.detach().numpy()[0])
        return results

    def save_model(self, model_filename):

        torch.save(self._model, model_filename)

    def load_model(self, model_filename):

        self._model = torch.load(model_filename)
        self._model.eval()
