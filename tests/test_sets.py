import torch

from graphchem.datasets import load_cn, load_lhv, load_mon, load_ron, load_ysi


# Test cases for loading property datasets
def test_loading_datasets():

    datasets = [
        load_cn,
        load_lhv,
        load_mon,
        load_ron,
        load_ysi,
    ]

    for dataset in datasets:

        smiles, prop = dataset()
        assert len(smiles) == len(prop)
        assert type(smiles) is list
        assert type(smiles[0]) is str
        assert type(prop) is torch.Tensor
        assert prop.dtype == torch.float32
