# GraphChem Documentation

[![UML Energy & Combustion Research Laboratory](https://sites.uml.edu/hunter-mack/files/2021/11/ECRL_final.png)](http://faculty.uml.edu/Hunter_Mack/)

[![GitHub version](https://badge.fury.io/gh/ecrl%2FGraphChem.svg)](https://badge.fury.io/gh/ecrl%2FGraphChem)
[![PyPI version](https://badge.fury.io/py/graphchem.svg)](https://badge.fury.io/py/graphchem)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/ecrl/GraphChem/master/LICENSE.txt)

**GraphChem** is an open source Python package for constructing graph-based machine learning models with a focus on fuel property prediction. 

Future plans for GraphChem include:

- Robust hyper-parameter and model architecture tuning runtimes

- Molecule visualization via RDKit

# Installation:

### Prerequisites:
- Have Python 3.11+ installed

### Method 1: pip
```
$ pip install graphchem
```

### Method 2: From Source
```
$ git clone https://github.com/ecrl/graphchem
$ cd graphchem
$ python setup.py install
```

If any errors occur when installing dependencies, namely with RDKit, PyTorch, or torch-geometric, visit their installation pages and follow the installation instructions: [RDKit](https://www.rdkit.org/docs/Install.html), [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

# Examples

To view some examples of how GraphChem can be used, head over to our [examples](https://github.com/ecrl/graphchem/tree/master/examples) folder on GitHub.

# Contributing, Reporting Issues and Other Support:

To contribute to GraphChem, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (Travis_Kessler@student.uml.edu).