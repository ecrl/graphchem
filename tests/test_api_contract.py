"""Lock public API signatures (design §8 / §11.1; blueprint T1.2)."""

from __future__ import annotations

import inspect
from importlib.metadata import PackageNotFoundError, version
from unittest.mock import patch

import pytest
import torch.nn.functional as F

import graphchem
import graphchem.data as graphchem_data
import graphchem.datasets as graphchem_datasets
import graphchem.nn as graphchem_nn
import graphchem.preprocessing as graphchem_preprocessing
import graphchem.version as graphchem_version
from graphchem import __version__
from graphchem.data import MoleculeDataset, MoleculeGraph
from graphchem.datasets import load_cn, load_lhv, load_mon, load_ron, load_ysi
from graphchem.nn import MoleculeGCN
from graphchem.preprocessing import (
    MoleculeEncoder,
    Tokenizer,
    atom_to_str,
    bond_to_str,
    get_ring_size,
    load_encoder,
)

_EXPECTED_ALL: dict[str, list[str]] = {
    "graphchem": ["__version__"],
    "graphchem.data": ["MoleculeGraph", "MoleculeDataset"],
    "graphchem.datasets": [
        "load_cn",
        "load_lhv",
        "load_mon",
        "load_ron",
        "load_ysi",
    ],
    "graphchem.nn": ["MoleculeGCN"],
    "graphchem.preprocessing": [
        "get_ring_size",
        "atom_to_str",
        "bond_to_str",
        "Tokenizer",
        "MoleculeEncoder",
        "load_encoder",
    ],
}

_ALL_MODULES = {
    "graphchem": graphchem,
    "graphchem.data": graphchem_data,
    "graphchem.datasets": graphchem_datasets,
    "graphchem.nn": graphchem_nn,
    "graphchem.preprocessing": graphchem_preprocessing,
}

_EMPTY = inspect.Parameter.empty

# Expected (name, default) in declaration order. Defaults only — not annotations.
# `self` is omitted; locks are for the caller-facing surface.
_EXPECTED_SIGNATURES: dict[object, list[tuple[str, object]]] = {
    MoleculeGraph.__init__: [
        ("atom_attr", _EMPTY),
        ("bond_attr", _EMPTY),
        ("connectivity", _EMPTY),
        ("target", None),
    ],
    MoleculeDataset.__init__: [
        ("graphs", _EMPTY),
    ],
    MoleculeGCN.__init__: [
        ("atom_vocab_size", _EMPTY),
        ("bond_vocab_size", _EMPTY),
        ("output_dim", _EMPTY),
        ("embedding_dim", 128),
        ("n_messages", 2),
        ("n_readout", 2),
        ("readout_dim", 64),
        ("p_dropout", 0.0),
        ("aggr", "add"),
        ("act_fn", F.softplus),
    ],
    MoleculeGCN.forward: [
        ("data", _EMPTY),
    ],
    MoleculeEncoder.__init__: [
        ("smiles", _EMPTY),
    ],
    MoleculeEncoder.encode: [
        ("smiles", _EMPTY),
    ],
    MoleculeEncoder.encode_many: [
        ("smiles", _EMPTY),
    ],
    MoleculeEncoder.save: [
        ("filename", _EMPTY),
    ],
    Tokenizer.__init__: [],
    Tokenizer.__call__: [
        ("item", _EMPTY),
    ],
    get_ring_size: [
        ("obj", _EMPTY),
        ("max_size", 12),
    ],
    atom_to_str: [
        ("atom", _EMPTY),
    ],
    bond_to_str: [
        ("bond", _EMPTY),
    ],
    load_encoder: [
        ("filename", _EMPTY),
    ],
    load_cn: [],
    load_lhv: [],
    load_mon: [],
    load_ron: [],
    load_ysi: [],
}


def _caller_params(obj: object) -> list[tuple[str, object]]:
    params = [p for p in inspect.signature(obj).parameters.values() if p.name != "self"]
    return [(p.name, p.default) for p in params]


@pytest.mark.parametrize(
    "obj",
    list(_EXPECTED_SIGNATURES),
    ids=lambda o: getattr(o, "__qualname__", getattr(o, "__name__", repr(o))),
)
def test_public_signature_names_and_defaults(obj: object) -> None:
    expected = _EXPECTED_SIGNATURES[obj]
    actual = _caller_params(obj)
    label = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
    assert actual == expected, (
        f"{label} signature drift:\n  expected={expected!r}\n  actual  ={actual!r}"
    )


@pytest.mark.parametrize("module_name", list(_EXPECTED_ALL))
def test_public_all_matches_frozen_surface(module_name: str) -> None:
    module = _ALL_MODULES[module_name]
    assert list(module.__all__) == _EXPECTED_ALL[module_name]
    for name in module.__all__:
        assert hasattr(module, name)


def test_version_matches_installed_packaging_metadata() -> None:
    assert isinstance(__version__, str)
    assert __version__
    assert __version__ == version("graphchem")


def test_version_fallback_reads_pyproject_when_not_installed() -> None:
    with patch(
        "graphchem.version.version",
        side_effect=PackageNotFoundError("graphchem"),
    ):
        resolved = graphchem_version._resolve_version()
    expected = graphchem_version._version_from_pyproject()
    assert expected is not None
    assert resolved == expected


def test_version_fallback_unknown_when_metadata_and_pyproject_missing() -> None:
    with (
        patch(
            "graphchem.version.version",
            side_effect=PackageNotFoundError("graphchem"),
        ),
        patch("graphchem.version._version_from_pyproject", return_value=None),
    ):
        resolved = graphchem_version._resolve_version()
    assert resolved == "0.0.0+unknown"
