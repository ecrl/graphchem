# API stability policy

This document is a **source-tree pointer** for contributors browsing without a
docs build.

**Canonical page:** Sphinx *API stability policy*
(`docs/source/api_stability.rst`), published on
[Read the Docs](https://graphchem.readthedocs.io/).

The frozen public surface, signature/behavioral contracts, versioning rules, and
oracle-parity gate are maintained in that Sphinx page. Keep this file and the
Sphinx page in sync when the contract changes.

**Deferred to 2.4.0** (not part of the 2.3.4 modernization release): optional
JSON/msgpack encoder serialization, CLI, pretrained weights, and a benchmark
harness. See `CHANGELOG.md` and the Sphinx versioning section.
