import pytest
import torch

from graphchem.datasets import load_cn, load_lhv, load_mon, load_ron, load_ysi

_LOADERS = [
    load_cn,
    load_lhv,
    load_mon,
    load_ron,
    load_ysi,
]


@pytest.mark.parametrize("loader", _LOADERS, ids=[fn.__name__ for fn in _LOADERS])
def test_loading_datasets(loader):
    smiles, prop = loader()

    assert isinstance(smiles, list)
    assert len(smiles) > 0
    assert all(isinstance(s, str) and s for s in smiles)

    assert isinstance(prop, torch.Tensor)
    assert prop.dtype == torch.float32
    assert prop.ndim == 2
    assert prop.shape == (len(smiles), 1)
    assert torch.isfinite(prop).all()
