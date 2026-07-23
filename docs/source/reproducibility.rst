Reproducibility
===============

Example notebooks under ``examples/`` are illustrative. To reduce run-to-run
variance when adapting them:

.. code-block:: python

   import numpy as np
   import torch

   torch.manual_seed(0)
   np.random.seed(0)

Notes
-----

* Encoding oracles in the test suite lock GraphChem behavior for a declared
  RDKit / GraphChem pair; they are regression anchors, not literature tables.
* Platform-specific wheels for RDKit, PyTorch, and PyTorch Geometric can still
  change floating-point results slightly. Prefer pinning or documenting the
  environment used for a published experiment.
* Dependency ranges are declared in ``pyproject.toml``; an optional
  ``environment.yml`` is not required for day-to-day development.
