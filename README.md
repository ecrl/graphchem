[![UML Energy & Combustion Research Laboratory](https://sites.uml.edu/hunter-mack/files/2021/11/ECRL_final.png)](http://faculty.uml.edu/Hunter_Mack/)

# GraphChem: Graph-based machine learning for chemical property prediction

[![GitHub version](https://badge.fury.io/gh/ecrl%2FGraphChem.svg)](https://badge.fury.io/gh/ecrl%2FGraphChem)
[![PyPI version](https://badge.fury.io/py/graphchem.svg)](https://badge.fury.io/py/graphchem)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/ecrl/graphchem/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/graphchem/badge/?version=latest)](https://graphchem.readthedocs.io/en/latest/?badge=latest)

**GraphChem** is an open source Python package for constructing graph-based machine learning models with a focus on fuel property prediction.

# Installation

### Prerequisites

- Python 3.10 or newer

### Method 1: pip

```
$ pip install graphchem
```

### Method 2: From source

```
$ git clone https://github.com/ecrl/graphchem
$ cd graphchem
$ python -m pip install .
```

If installation fails for RDKit, PyTorch, or torch-geometric, follow the vendor guides: [RDKit](https://www.rdkit.org/docs/Install.html), [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

# Documentation

- [Read the Docs (Sphinx)](https://graphchem.readthedocs.io/en/latest/) — installation, quickstart, API, datasets, stability policy
- [API_STABILITY.md](API_STABILITY.md) — source-tree pointer to the frozen public surface
- [CHANGELOG.md](CHANGELOG.md)

# Examples

Notebooks are under [`examples/`](https://github.com/ecrl/graphchem/tree/master/examples) on GitHub. See also the tutorials page in the docs.

For reproducible runs, set seeds before training or evaluation, for example:

```python
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
```

Exact numeric results can still vary across platforms and dependency wheel builds (RDKit / PyTorch / PyG).

# Contributing and support

- Contribute via pull request; follow [CONTRIBUTING.md](CONTRIBUTING.md)
- Report bugs and feature requests with [GitHub Issues](https://github.com/ecrl/graphchem/issues)
- Security reports: [SECURITY.md](SECURITY.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

For additional questions, contact Travis Kessler (`travis.j.kessler@gmail.com`).
