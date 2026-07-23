API reference
=============

Autodocumentation of the frozen public surface. Behavioral contracts are
documented in :doc:`api_stability`.

Package version
---------------

.. autodata:: graphchem.__version__
   :annotation:

``graphchem.data``
------------------

.. autoclass:: graphchem.data.MoleculeGraph
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: graphchem.data.MoleculeDataset
   :members:
   :special-members: __init__
   :show-inheritance:

``graphchem.datasets``
----------------------

.. autofunction:: graphchem.datasets.load_cn
.. autofunction:: graphchem.datasets.load_lhv
.. autofunction:: graphchem.datasets.load_mon
.. autofunction:: graphchem.datasets.load_ron
.. autofunction:: graphchem.datasets.load_ysi

``graphchem.nn``
----------------

.. autoclass:: graphchem.nn.MoleculeGCN
   :members:
   :special-members: __init__
   :show-inheritance:

``graphchem.preprocessing``
---------------------------

.. autofunction:: graphchem.preprocessing.get_ring_size
.. autofunction:: graphchem.preprocessing.atom_to_str
.. autofunction:: graphchem.preprocessing.bond_to_str

.. autoclass:: graphchem.preprocessing.Tokenizer
   :members:
   :special-members: __init__, __call__

.. autoclass:: graphchem.preprocessing.MoleculeEncoder
   :members:
   :special-members: __init__

.. autofunction:: graphchem.preprocessing.load_encoder
