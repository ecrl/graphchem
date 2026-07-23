API stability policy
====================

This page states the public API contract for **graphchem** and how version
numbers relate to compatible change.

A markdown copy for contributors browsing the source tree without a docs build
is kept at ``API_STABILITY.md`` in the repository root.

GraphChem builds graph neural network models for molecular (fuel) property
prediction from SMILES, using RDKit for perception and PyTorch Geometric for
graphs and message passing.

Frozen public surface
---------------------

The following imports are the supported surface and must remain drop-in
compatible within the ``2.3.x`` / ``2.4.x`` compatibility series:

.. code-block:: python

   from graphchem import __version__

   from graphchem.data import MoleculeGraph, MoleculeDataset

   from graphchem.datasets import (
       load_cn, load_lhv, load_mon, load_ron, load_ysi,
   )

   from graphchem.nn import MoleculeGCN

   from graphchem.preprocessing import (
       get_ring_size,
       atom_to_str,
       bond_to_str,
       Tokenizer,
       MoleculeEncoder,
       load_encoder,
   )

Parameter **names** and **defaults** for these callables and constructors are
frozen. Additive optional keyword arguments may be introduced when they do not
change existing call sites. Renaming exports, making new arguments required,
changing primary return container types or shapes, or changing documented
exception types for existing failure modes requires an explicit, versioned
compatibility decision.

Top-level re-exports of submodule symbols (beyond ``__version__``) are **not**
required for compatibility. Callers should import from the submodules above.

Signature contracts
-------------------

Keyword names and defaults must remain compatible (additive optional kwargs
only):

.. code-block:: python

   MoleculeGraph(atom_attr, bond_attr, connectivity, target=None)
   MoleculeDataset(graphs)

   MoleculeGCN(
       atom_vocab_size,
       bond_vocab_size,
       output_dim,
       embedding_dim=128,
       n_messages=2,
       n_readout=2,
       readout_dim=64,
       p_dropout=0.0,
       aggr="add",
       act_fn=...,  # default: torch.nn.functional.softplus
   )

   MoleculeEncoder(smiles)          # smiles: list[str]
   MoleculeEncoder.encode(smiles)   # -> (atom_tensor, bond_tensor, connectivity)
   MoleculeEncoder.encode_many(...)
   MoleculeEncoder.save(filename)
   load_encoder(filename)

   Tokenizer.__call__(...)
   # Tokenizer.train, Tokenizer.unknown, Tokenizer.vocab_size

   get_ring_size(obj, max_size=12)
   atom_to_str(atom)
   bond_to_str(bond)

   load_cn()
   load_lhv()
   load_mon()
   load_ron()
   load_ysi()

Behavioral contracts
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Contract
     - Detail
   * - Bond duplication
     - ``MoleculeEncoder.encode()`` produces ``2 × num_bonds`` bond features
   * - Target shape
     - ``MoleculeGraph`` enforces ``y.shape[0] == 1``
   * - Featurizer strings
     - ``atom_to_str`` / ``bond_to_str`` tuple-string format drives vocabularies
   * - Encoder persistence
     - ``MoleculeEncoder.save`` / ``load_encoder`` remain pickle-compatible
   * - GCN outputs
     - ``MoleculeGCN.forward(data)`` returns ``(out, out_atom, out_bond)``
   * - ``n_readout=0``
     - Returns pooled embeddings without a readout MLP
   * - Dataset loaders
     - ``load_*()`` → ``(list[str], torch.Tensor)`` with stable column semantics
   * - Invalid SMILES
     - ``MoleculeEncoder`` raises ``ValueError`` on unparsable SMILES

Versioning
----------

1. **Patch / compatible minor (``2.3.x`` / ``2.4.x``):** bug fixes that restore
   documented behavior; tooling; documentation; dependency range changes proven
   by encoding/oracle tests; additive optional kwargs with defaults matching
   today; clearer error messages of the same exception type. Featurizer string
   schemas and pickle encoder load must continue to work.
2. **Modernization release (``2.3.4``):** the API-stable modernization program
   ships as **``2.3.4``** at the end of Phase 5. Phases 1–4 remain on tip
   ``2.3.3`` until that bump.
3. **Later minor (``2.4.x``):** reserved for additive, still API-compatible work
   after ``2.3.4``. Explicitly deferred from the ``2.3.4`` modernization release:

   * optional JSON/msgpack encoder serialization (pickle remains)
   * optional CLI entry points
   * optional pretrained model weights
   * benchmark harness

4. **Major (``3.0.0``):** featurizer string schema changes that break existing
   encoders; removal of pickle load; changes to ``forward`` return arity/shapes;
   dataset content revisions that change labels for the same SMILES.

Oracle parity as a release gate
-------------------------------

Regression oracles in the test suite (fixed SMILES encodings, encoder
round-trips, and end-to-end smoke paths) are part of the compatibility contract
for the ``2.3.x`` / ``2.4.x`` series.

Fixture values are self-consistency anchors for the declared RDKit and GraphChem
versions, not independent literature tables, unless explicitly expanded later.

Runtime prerequisites
---------------------

GraphChem depends on **RDKit**, **NumPy**, **PyTorch**, and **PyTorch Geometric**.
Platform-specific wheel installation may require following the upstream install
guides linked from :doc:`installation`.
