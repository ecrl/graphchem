"""MoleculeEncoder encoding oracles and contracts (design §11.1; T1.3–T1.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from graphchem.preprocessing import MoleculeEncoder, load_encoder

_INVALID_SMILES = "not_a_smiles"

_FIXTURES = Path(__file__).parent / "fixtures"
_ORACLE_PATH = _FIXTURES / "encoding_oracles.json"


@pytest.fixture(scope="module")
def encoding_oracles() -> dict:
    with _ORACLE_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_encoding_oracle_fixture_schema(encoding_oracles: dict) -> None:
    assert encoding_oracles["schema_version"] == 1
    assert encoding_oracles["train_smiles"] == ["CC", "CCO", "c1ccccc1"]
    molecules = encoding_oracles["molecules"]
    assert set(molecules) == {"CC", "CCO", "c1ccccc1"}
    for smi, payload in molecules.items():
        assert "atoms" in payload and "bonds" in payload and "connectivity" in payload
        assert len(payload["connectivity"]) == 2
        # Bond duplication: bond feature count is 2 × number of undirected bonds.
        n_directed = len(payload["bonds"])
        assert n_directed == len(payload["connectivity"][0])
        assert n_directed == len(payload["connectivity"][1])
        assert n_directed % 2 == 0, f"{smi}: expected even bond-feature count"


@pytest.mark.parametrize("smiles", ["CC", "CCO", "c1ccccc1"])
def test_encode_matches_golden_oracle(encoding_oracles: dict, smiles: str) -> None:
    train_smiles = encoding_oracles["train_smiles"]
    expected = encoding_oracles["molecules"][smiles]
    encoder = MoleculeEncoder(train_smiles)

    atoms, bonds, connectivity = encoder.encode(smiles)

    assert atoms.dtype in (torch.int32, torch.int64, torch.int)
    assert torch.equal(atoms, torch.tensor(expected["atoms"]).type_as(atoms))
    assert torch.equal(bonds, torch.tensor(expected["bonds"]).type_as(bonds))
    assert torch.equal(
        connectivity,
        torch.tensor(expected["connectivity"]).type_as(connectivity),
    )


def test_ethane_bond_duplication(encoding_oracles: dict) -> None:
    """CC has one undirected bond → two directed bond features."""
    expected = encoding_oracles["molecules"]["CC"]
    assert expected["atoms"] == [2, 2]
    assert expected["bonds"] == [2, 2]
    assert expected["connectivity"] == [[0, 1], [1, 0]]


def test_encoder_pickle_round_trip_preserves_encodings(
    encoding_oracles: dict, tmp_path: Path
) -> None:
    """save/load_encoder pickle round-trip keeps vocab and fixture encodings."""
    train_smiles = encoding_oracles["train_smiles"]
    encoder = MoleculeEncoder(train_smiles)
    path = tmp_path / "encoder.pkl"
    encoder.save(str(path))

    loaded = load_encoder(str(path))
    assert loaded.vocab_sizes == encoder.vocab_sizes

    for smiles, expected in encoding_oracles["molecules"].items():
        atoms, bonds, connectivity = loaded.encode(smiles)
        assert torch.equal(atoms, torch.tensor(expected["atoms"]).type_as(atoms))
        assert torch.equal(bonds, torch.tensor(expected["bonds"]).type_as(bonds))
        assert torch.equal(
            connectivity,
            torch.tensor(expected["connectivity"]).type_as(connectivity),
        )


def test_molecule_encoder_init_invalid_smiles_raises_value_error() -> None:
    with pytest.raises(ValueError):
        MoleculeEncoder([_INVALID_SMILES])


def test_molecule_encoder_encode_invalid_smiles_raises_value_error(
    encoding_oracles: dict,
) -> None:
    encoder = MoleculeEncoder(encoding_oracles["train_smiles"])
    with pytest.raises(ValueError):
        encoder.encode(_INVALID_SMILES)
