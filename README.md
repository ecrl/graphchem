# GraphChem: Graph-based machine learning for chemical property prediction

[![GitHub version](https://badge.fury.io/gh/tjkessler%2FGraphChem.svg)](https://badge.fury.io/gh/tjkessler%2FGraphChem)
[![PyPI version](https://badge.fury.io/py/graphchem.svg)](https://badge.fury.io/py/graphchem)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/tjkessler/GraphChem/master/LICENSE.txt)

**GraphChem** is an open source Python package for constructing graph-based machine learning models with a focus on chemical property prediction. 

Future plans for GraphChem include:
- Rendering bond/atom contribution as an overlay on an image of the compound

# Installation:

### Prerequisites:
- Have Python 3.5+ installed
- Have [RDKit](https://www.rdkit.org/docs/Install.html) installed (using Conda environments is highly recommended)

### Method 1: pip
```
$ pip install graphchem
```

If any installation issues occur with PyTorch or torch-dependent packages (notably torch-geometric), visit their installation pages and follow the installation instructions: [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Method 2: From Source
```
$ git clone https://github.com/tjkessler/graphchem
$ cd graphchem
$ python setup.py install
```

# Usage:

API documentation is coming in the future! In the meantime, take a look at some [examples](https://github.com/tjkessler/GraphChem/tree/master/examples).

# Contributing, Reporting Issues and Other Support:

To contribute to GraphChem, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com).
