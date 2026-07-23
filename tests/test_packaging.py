"""Wheel package-data checks (blueprint T1.13).

Skips unless a built wheel is available via ``GRAPHCHEM_WHEEL`` or
``dist/graphchem-*.whl``. Run ``python -m build`` first to exercise these.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STATIC_CSVS = (
    "graphchem/datasets/static/cn.csv",
    "graphchem/datasets/static/lhv.csv",
    "graphchem/datasets/static/mon.csv",
    "graphchem/datasets/static/ron.csv",
    "graphchem/datasets/static/ysi.csv",
)


def _resolve_wheel() -> Path | None:
    env = os.environ.get("GRAPHCHEM_WHEEL")
    if env:
        path = Path(env).expanduser().resolve()
        return path if path.is_file() else None
    wheels = sorted(
        (_REPO_ROOT / "dist").glob("graphchem-*.whl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return wheels[0] if wheels else None


@pytest.fixture(scope="module")
def wheel_path() -> Path:
    path = _resolve_wheel()
    if path is None:
        pytest.skip("No wheel found; set GRAPHCHEM_WHEEL or run `python -m build`")
    return path


def test_wheel_contains_static_csv_package_data(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
    missing = [name for name in _STATIC_CSVS if name not in names]
    assert not missing, f"Wheel {wheel_path.name} missing package data: {missing}"
