import torch
import torch_geometric.data as gdata
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from re import compile

from .preprocessing import CompoundEncoder
from .models import CompoundGCN, LRDecayLinear, CallbackOperator, Validator

_PT_RE = compile(r'^.*\.pt$')
_ENC_RE = compile(r'^.*\.enc$')


class CompoundOperator(object):

    def __init__(self, device: str = None):
        """
        CompoundOperator(object): handles data pre-processing, model training,
        model saving/recall, using a model on new data

        Args:
            device (str, optional): torch.device to run ops on, defaults to
                `cuda:0` if available else `cpu`
        """

        if device is None:
            self._device = torch.device('cuda:0' if torch.cuda.is_available()
                                        else 'cpu')
        else:
            self._device = device
        self._model = None
        self._ce = None

    def train(self, smiles: List[str], target: List[List[float]],
              model_config: dict = None, valid_size: float = 0.2,
              valid_epoch_iter: int = 1, valid_patience: int = 16,
              batch_size: int = 1, lr: float = 0.001, lr_decay: float = 0.0,
              epochs: int = 128, verbose: int = 0, random_state: int = None,
              **kwargs) -> Tuple[List[float], List[float]]:
        """
        Trains a CompoundCGN using supplied SMILES strings, target values

        Args:
            smiles (list[str]): list of SMILES strings, one per compound
            target (list[list[float]]): list of target values, shape
                [n_samples, n_targets], one per compound
            model_filename (str, optional): if not `None`, saves the trained
                model to this filename/path
            model_config (dict, optional): if not supplied, uses default model
            architecture:
                {
                    'n_messages': 1,
                    'n_hidden': 1,
                    'hidden_dim': 32,
                    'dropout': 0.00
                }
            valid_size (float, optional): proportion of training set used for
                periodic validation, default = 0.2
            valid_epoch_iter (int, optional): validation set performance is
                measured every `this` epochs, default = 1 epochs
            valid_patience (int, optional): if lower validation set loss not
                encountered after `this` many epochs, terminate to avoid
                overfitting, default = 16
            batch_size (int, optional): size of each batch during training,
                default = 1
            lr (float, optional): learning rate for Adam opt, default = 0.001
            lr_decay (float, optional): linear rate of decay of learning rate
                per epoch, default = 0.0
            epochs (int, optional): number of training epochs, default = 128
            verbose (int, optional): training and validation loss printed to
                console every `this` epochs, default = 0 (no printing)
            random_state (int, optional): if not `None`, seeds validation
                subset randomized selection with this value
            **kwargs: additional arguments passed to torch.optim.Adam

        Returns:
            tuple[list[float], list[float]]: (training losses, validation
                losses) over all training epochs
        """

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
            a, b, c = self._ce.encode(smi)
            data.append(gdata.Data(
                x=a,
                edge_index=c,
                edge_attr=b,
                y=torch.tensor(target[idx]).type(torch.float)
            ).to(self._device))

        # Split data into training, validation subsets
        data_train, data_valid = train_test_split(
            data, test_size=valid_size, random_state=random_state
        )
        loader_train = gdata.DataLoader(
            data_train,
            batch_size=batch_size,
            shuffle=True
        )
        loader_valid = gdata.DataLoader(
            data_valid,
            batch_size=batch_size,
            shuffle=True
        )

        # Create model
        if model_config is None:
            self._model = CompoundGCN(
                self._ce.ATOM_DIM,
                self._ce.BOND_DIM,
                len(target[0])
            )
        else:
            self._model = CompoundGCN(
                self._ce.ATOM_DIM,
                self._ce.BOND_DIM,
                len(target[0]),
                model_config['n_messages'],
                model_config['n_hidden'],
                model_config['hidden_dim'],
                model_config['dropout']
            )
        self._model.to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, **kwargs)

        # Setup callbacks
        CBO = CallbackOperator()
        _lrdecay = LRDecayLinear(lr, lr_decay, optimizer)
        _validator = Validator(
            loader_valid,
            self._model,
            valid_epoch_iter,
            valid_patience
        )
        CBO.add_cb(_lrdecay)
        CBO.add_cb(_validator)

        # Record loss for return
        train_losses = []
        valid_losses = []

        # TRAIN BEGIN
        CBO.on_train_begin()

        # Begin training loop
        for epoch in range(epochs):

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
                pred, _, _ = self._model(batch)
                target = batch.y

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

            if verbose > 0:
                if epoch % verbose == 0:
                    print('Epoch: {} | Train Loss: {} | Valid Loss: {}'.format(
                          epoch, train_loss, _validator._most_recent_loss))

            train_losses.append(train_loss)
            valid_losses.append(_validator._most_recent_loss.detach().item())

        # TRAIN END
        CBO.on_train_end()

        return (train_losses, valid_losses)

    def use(self, smiles: List[str],
            model_filename: str = None) -> List[List[float]]:
        """
        Uses a pre-trained CompoundGCN, either trained in-session or recalled
        from a file, for use on new data

        Args:
            smiles (list[str]): SMILES strings to predict for
            model_filename (str, optional): filename/path of model to load,
                default = None (model trained in-session used)

        Returns:
            list[list[float]]: predicted values of shape [n_samples, n_targets]
        """

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
            a, b, c = self._ce.encode(smi)
            data.append(gdata.Data(
                x=a,
                edge_index=c,
                edge_attr=b
            ).to(self._device))
        loader_test = gdata.DataLoader(
            data,
            batch_size=1,
            shuffle=False
        )

        # Get results
        results = []
        for batch in loader_test:
            res, _, _ = self._model(batch)
            results.append(res.detach().numpy().tolist()[0])
        return results

    def save_model(self, model_filename: str, encoder_filename: str):
        """
        Saves model, and the necessary encoder, to two files for later use

        Args:
            model_filename (str): filename/path to save model, `.pt` extension
            encoder_filename (str): filename/path to save encoder, `.enc`
                extension
        """

        if _PT_RE.match(model_filename) is None:
            raise ValueError('model_filename must have `.pt` extension')
        if _ENC_RE.match(encoder_filename) is None:
            raise ValueError(('encoder_filename must have a `.enc` extension'))

        torch.save(self._model, model_filename)
        torch.save(self._ce, model_filename.replace('.pt', '.enc'))

    def load_model(self, model_filename: str, encoder_filename: str):
        """
        Loads a pre-trained CompoundGCN and its encoder for use in-session

        Args:
            model_filename (str): filename/path of pre-trained model, `.pt`
                extension
            encoder_filename (str): filename/path of model's encoder, `.enc`
                extension
        """

        if _PT_RE.match(model_filename) is None:
            raise ValueError('model_filename must have `.pt` extension')

        self._model = torch.load(model_filename)
        self._model.eval()
        self._ce = torch.load(model_filename.replace('.pt', '.enc'))
