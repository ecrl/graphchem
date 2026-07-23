# Test fixtures (GraphChem encoding self-consistency)

JSON oracles used by the GraphChem test suite as **regression anchors** for
`MoleculeEncoder.encode` under a fixed training corpus. Expected atom/bond/
connectivity integer tensors are self-consistency anchors for the declared
GraphChem + RDKit stack at capture time. They are **not** independent literature
reference tables.

## Contents

| File | Format | Role |
|------|--------|------|
| `encoding_oracles.json` | JSON | Golden encodings for `CC`, `CCO`, `c1ccccc1` |

## Corpus

Encodings depend on tokenizer vocabulary assignment, which depends on the
**order** of SMILES passed to `MoleculeEncoder(...)`. Oracles must be checked
with the exact `train_smiles` list stored in the JSON (`["CC", "CCO", "c1ccccc1"]`).

| SMILES | Why included |
|--------|----------------|
| `CC` | Single bond; bond-feature duplication → `2 × num_bonds` |
| `CCO` | Multi-bond chain with heteroatom |
| `c1ccccc1` | Aromatic ring; multi-bond |

## Provenance

Captured from a local editable install of this repository before the RDKit
package-name migration (`rdkit-pypi` → `rdkit`). See the `capture` object inside
`encoding_oracles.json` for `graphchem_version`, `rdkit_module_version`, and
distribution metadata recorded at generation time.

## License and redistribution

Fixture values are synthetic regression artifacts derived from public SMILES
strings for software testing. No third-party dataset tables are embedded.

## Limitations

- Integer token IDs are **not** chemically meaningful identifiers; they are
  vocabulary indices for a specific training corpus order.
- Changing RDKit versions, featurizer string schemas (`atom_to_str` /
  `bond_to_str`), or bond-duplication behavior may invalidate oracles and
  requires an explicit compatibility decision (see `API_STABILITY.md`).
- Values are not comparable across other molecular featurizers or graph libraries.
