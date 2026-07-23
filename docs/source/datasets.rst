Bundled datasets
================

GraphChem ships five curated fuel-property CSVs under
``graphchem/datasets/static/``. Load them with the ``graphchem.datasets``
helpers. These tables are for examples and reproducible experiments in this
package; they are **not** a live external database.

Common fields
-------------

* **Format:** CSV with a ``SMILES`` column and one property column.
* **Loader return type:** ``tuple[list[str], torch.Tensor]`` where the tensor
  has shape ``(n, 1)`` and dtype ``float32``.
* **Package license:** MIT (code). Dataset provenance is separate — see each
  card.
* **Limitations:** Bundled snapshot only; labels are not revalidated here;
  invalid or unusual SMILES may still be present as historically shipped.

Provenance for all five sets is **Unknown** pending maintainer confirmation
(design Q5). Do not treat the numeric labels as citable literature values
without an external reference.

Cetane number (CN)
------------------

* **Identifier:** ``cn``
* **Loader:** :func:`graphchem.datasets.load_cn`
* **Package data:** ``graphchem/datasets/static/cn.csv``
* **Columns:** ``SMILES``, ``CN``
* **Rows:** 460
* **Property:** Cetane number
* **Units:** Unknown (values as stored in the CSV)
* **Provenance / primary literature:** Unknown — omitted for now
* **Access / license notes:** Distributed with the GraphChem package; confirm
  redistribution terms with maintainers before republishing the table alone

Lower heating value (LHV)
-------------------------

* **Identifier:** ``lhv``
* **Loader:** :func:`graphchem.datasets.load_lhv`
* **Package data:** ``graphchem/datasets/static/lhv.csv``
* **Columns:** ``SMILES``, ``LHV``
* **Rows:** 388
* **Property:** Lower heating value
* **Units:** Unknown (values as stored in the CSV)
* **Provenance / primary literature:** Unknown — omitted for now
* **Access / license notes:** Distributed with the GraphChem package; confirm
  redistribution terms with maintainers before republishing the table alone

Motor octane number (MON)
-------------------------

* **Identifier:** ``mon``
* **Loader:** :func:`graphchem.datasets.load_mon`
* **Package data:** ``graphchem/datasets/static/mon.csv``
* **Columns:** ``SMILES``, ``MON``
* **Rows:** 308
* **Property:** Motor octane number
* **Units:** Unknown (octane number scale as stored)
* **Provenance / primary literature:** Unknown — omitted for now
* **Access / license notes:** Distributed with the GraphChem package; confirm
  redistribution terms with maintainers before republishing the table alone

Research octane number (RON)
----------------------------

* **Identifier:** ``ron``
* **Loader:** :func:`graphchem.datasets.load_ron`
* **Package data:** ``graphchem/datasets/static/ron.csv``
* **Columns:** ``SMILES``, ``RON``
* **Rows:** 308
* **Property:** Research octane number
* **Units:** Unknown (octane number scale as stored)
* **Provenance / primary literature:** Unknown — omitted for now
* **Access / license notes:** Distributed with the GraphChem package; confirm
  redistribution terms with maintainers before republishing the table alone

Yield sooting index (YSI)
-------------------------

* **Identifier:** ``ysi``
* **Loader:** :func:`graphchem.datasets.load_ysi`
* **Package data:** ``graphchem/datasets/static/ysi.csv``
* **Columns:** ``SMILES``, ``YSI``
* **Rows:** 558
* **Property:** Yield sooting index
* **Units:** Unknown (YSI scale as stored; includes negative values)
* **Provenance / primary literature:** Unknown — omitted for now
* **Access / license notes:** Distributed with the GraphChem package; confirm
  redistribution terms with maintainers before republishing the table alone
