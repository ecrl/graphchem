Quickstart
==========

This page shows a minimal encode → graph → GCN forward path using a tiny SMILES
subset. For full training notebooks, see :doc:`tutorials`.

Load data and encode SMILES
---------------------------

.. code-block:: python

   import torch
   from graphchem.datasets import load_cn
   from graphchem.preprocessing import MoleculeEncoder

   smiles, targets = load_cn()
   smiles = smiles[:8]
   targets = targets[:8]

   encoder = MoleculeEncoder(smiles)
   atom_vocab, bond_vocab = encoder.vocab_sizes
   encodings = encoder.encode_many(smiles)

Build graphs and a model
------------------------

.. code-block:: python

   from torch_geometric.loader import DataLoader

   from graphchem.data import MoleculeDataset, MoleculeGraph
   from graphchem.nn import MoleculeGCN

   graphs = [
       MoleculeGraph(atoms, bonds, connectivity, targets[i])
       for i, (atoms, bonds, connectivity) in enumerate(encodings)
   ]
   loader = DataLoader(MoleculeDataset(graphs), batch_size=4, shuffle=False)

   model = MoleculeGCN(
       atom_vocab_size=atom_vocab,
       bond_vocab_size=bond_vocab,
       output_dim=1,
       embedding_dim=32,
       n_messages=1,
       n_readout=1,
       readout_dim=16,
       p_dropout=0.0,
   )
   model.eval()

   batch = next(iter(loader))
   out, out_atom, out_bond = model(batch)
   assert out.shape[0] == batch.num_graphs
   assert torch.isfinite(out).all()

``MoleculeGCN.forward`` always returns ``(out, out_atom, out_bond)``. With
``n_readout=0``, ``out`` is the pooled embedding (no MLP head).

Next steps
----------

* :doc:`api` — public classes and functions
* :doc:`api_stability` — compatibility contracts
* :doc:`tutorials` — example notebooks on GitHub
* :doc:`reproducibility` — seeds and environment notes
