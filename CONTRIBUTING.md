# Contributing to GraphChem

Thanks for contributing. This guide covers the local workflow for code changes.
Public API stability is documented in [`API_STABILITY.md`](API_STABILITY.md).

## Development setup

```bash
git clone https://github.com/ecrl/graphchem
cd graphchem
python -m pip install -e ".[dev]"
pre-commit install
```

Optional docs dependencies (Sphinx):

```bash
python -m pip install -e ".[docs]"
```

Build HTML docs locally:

```bash
sphinx-build -W -b html docs/source docs/_build/html
```

## Local checks

Run these before opening a pull request:

```bash
ruff check src tests
ruff format --check src tests
pytest tests/ --cov=graphchem --cov-report=term-missing
```

Pre-commit runs ruff on staged files:

```bash
pre-commit run --all-files
```

When packaging changes are in scope:

```bash
python -m build
```

## Pull requests

- Keep changes focused; do not break the frozen public API without discussion.
- Add or update tests when behavior changes.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.
- Link related issues in the PR description.

## Releases

Maintainers publish by cutting a GitHub Release. The
`Publish GraphChem to PyPI` workflow builds with `python -m build` and uploads
via PyPI trusted publishing (OIDC). See [`SECURITY.md`](SECURITY.md).

## Reporting issues

Use GitHub Issues. Include OS, Python version, package version (`import graphchem; print(graphchem.__version__)`), and a minimal reproduction when reporting bugs.

Security-sensitive reports: see [`SECURITY.md`](SECURITY.md).
