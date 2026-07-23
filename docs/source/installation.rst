Installation
============

Prerequisites
-------------

* Python 3.10 or newer

Install from PyPI
-----------------

.. code-block:: bash

   pip install graphchem

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/ecrl/graphchem
   cd graphchem
   python -m pip install .

Platform-specific dependencies
------------------------------

If installation fails for RDKit, PyTorch, or PyTorch Geometric, follow the
vendor install guides:

* `RDKit <https://www.rdkit.org/docs/Install.html>`_
* `PyTorch <https://pytorch.org/get-started/locally/>`_
* `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_

Development extras
------------------

.. code-block:: bash

   python -m pip install -e ".[dev]"
   python -m pip install -e ".[docs]"
