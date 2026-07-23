# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Deferred to 2.4.0

See the `[2.3.4]` section for additives intentionally held back from the
modernization release.

## [2.3.4] - 2026-07-22

API-stable modernization release. Public callables and behavioral contracts are
unchanged; see `API_STABILITY.md` / Sphinx *API stability policy*.

### Changed

- Package layout migrated to `src/graphchem/`.
- Runtime dependency: `rdkit-pypi` replaced by `rdkit` with compatible version ranges for `numpy`, `torch`, and `torch-geometric`.
- Supported Python versions: `>=3.10` (CI matrix 3.10–3.13).
- RDKit valence access uses a compatibility helper (`GetValence`) while preserving featurizer outputs.
- Typing honesty for `MoleculeGCN` (including `output_dim: int | None`); package ships `py.typed`.
- PyPI publish workflow uses trusted publishing (OIDC).

### Added

- API contract suite, encoding oracles, e2e pipeline smoke tests, and packaging wheel checks.
- Coverage gate (fail-under 90%), ruff + pre-commit, Dependabot, and PR/push CI with lint, docs, and test matrix.
- Sphinx + Furo documentation (quickstart, tutorials, dataset cards, API autodoc, reproducibility notes).
- Governance files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CITATION.cff, CHANGELOG, issue/PR templates.
- `API_STABILITY.md` (source-tree pointer to the Sphinx stability page).

### Fixed

- `__version__` resolution for editable / non-installed workflows (metadata → `pyproject.toml` → unknown sentinel).

### Deferred to 2.4.0

- Optional JSON/msgpack encoder serialization (pickle `load_encoder` remains)
- Optional CLI entry points (encode / train / predict)
- Optional pretrained model weights
- Benchmark harness

## [2.3.3] - 2025-02-13

Baseline tip prior to the API-stable modernization program.

[Unreleased]: https://github.com/ecrl/graphchem/compare/2.3.4...HEAD
[2.3.4]: https://github.com/ecrl/graphchem/compare/2.3.3...2.3.4
[2.3.3]: https://github.com/ecrl/graphchem/releases/tag/2.3.3
