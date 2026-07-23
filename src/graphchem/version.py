"""Package version resolved from installed metadata or local pyproject.toml."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _version_from_pyproject() -> str | None:
    """Read ``[project].version`` from the nearest ``pyproject.toml``, if present."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        if not candidate.is_file():
            continue
        try:
            with candidate.open("rb") as fh:
                data = tomllib.load(fh)
            value = data.get("project", {}).get("version")
            if isinstance(value, str) and value:
                return value
        except (OSError, tomllib.TOMLDecodeError, AttributeError, TypeError):
            return None
    return None


def _resolve_version() -> str:
    try:
        return version("graphchem")
    except PackageNotFoundError:
        fallback = _version_from_pyproject()
        if fallback is not None:
            return fallback
        return "0.0.0+unknown"


__version__ = _resolve_version()
