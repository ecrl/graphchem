Tutorials and examples
======================

Executable notebooks live in the repository ``examples/`` directory on GitHub:

https://github.com/ecrl/graphchem/tree/master/examples

Property prediction
-------------------

These notebooks load a bundled dataset, encode SMILES, and train or evaluate a
``MoleculeGCN``:

* `predict_cn.ipynb <https://github.com/ecrl/graphchem/blob/master/examples/predict_cn.ipynb>`_ — cetane number
* `predict_lhv.ipynb <https://github.com/ecrl/graphchem/blob/master/examples/predict_lhv.ipynb>`_ — lower heating value
* `predict_mon.ipynb <https://github.com/ecrl/graphchem/blob/master/examples/predict_mon.ipynb>`_ — motor octane number
* `predict_ron.ipynb <https://github.com/ecrl/graphchem/blob/master/examples/predict_ron.ipynb>`_ — research octane number
* `predict_ysi.ipynb <https://github.com/ecrl/graphchem/blob/master/examples/predict_ysi.ipynb>`_ — yield sooting index

Comparison experiments
----------------------

Additional training notebooks and CSV splits under ``examples/comparison/``:

* `train_cn.ipynb <https://github.com/ecrl/graphchem/blob/master/examples/comparison/train_cn.ipynb>`_
* `train_mon.ipynb <https://github.com/ecrl/graphchem/blob/master/examples/comparison/train_mon.ipynb>`_
* `train_ron.ipynb <https://github.com/ecrl/graphchem/blob/master/examples/comparison/train_ron.ipynb>`_

Extras
------

Install optional example dependencies with:

.. code-block:: bash

   pip install -e ".[examples]"
