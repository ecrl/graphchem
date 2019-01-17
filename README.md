# GraphChem: Graph representations of molecules derived from SMILES strings

[![GitHub version](https://badge.fury.io/gh/tjkessler%2FGraphChem.svg)](https://badge.fury.io/gh/tjkessler%2FGraphChem)
[![PyPI version](https://badge.fury.io/py/graphchem.svg)](https://badge.fury.io/py/graphchem)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/tjkessler/GraphChem/master/LICENSE.txt)

**GraphChem** is an open source Python package for creating graph and vector representations of molecules from their simplified molecular-input line-entry system (SMILES) strings.

Future plans for GraphChem include:
- Implementing graph neural networks (GNN's) for molecular property prediction
- Rendering molecular structures as images

# Installation:

### Prerequisites:
- Have Python 3.X installed

### Method 1: pip
- **pip install graphchem**

### Method 2: From Source
- Download the GraphChem repository, navigate to the download location and execute **"python setup.py install"**

There are currently no dependencies for GraphChem.

# Usage:

Import the Graph object and initialize a Graph using a SMILES string. For this example, we will use 2,5-dimethylfuran:

```python
from graphchem import Graph
g = Graph('CC1=CC=C(C)O1')
```

Note: only molecules containing carbon and/or oxygen atoms are currently supported.

If we print the Graph, we can display each atom and its bonds:

```python
>>> print(g)
ID      Atom    Connections
0       C       [(1, 'Single')]
1       C       [(0, 'Single'), (2, 'Double'), (6, 'Single')]
2       C       [(1, 'Double'), (3, 'Single')]
3       C       [(2, 'Single'), (4, 'Double')]
4       C       [(3, 'Double'), (5, 'Single'), (6, 'Single')]
5       C       [(4, 'Single')]
6       O       [(4, 'Single'), (1, 'Single')]
```

Tuples in each atom's list of connections represent the ID of an atom it is bonded to and the bond type.

The Graph's pack() method returns vector representations for each atom:

```python
>>> for atom in g.pack():
>>>     print(atom)
(1, 0, 1, 0, 0, 0, 0)
(1, 0, 2, 1, 0, 0, 0)
(1, 0, 1, 1, 0, 0, 0)
(1, 0, 1, 1, 0, 0, 0)
(1, 0, 2, 1, 0, 0, 0)
(1, 0, 1, 0, 0, 0, 0)
(0, 1, 2, 0, 0, 0, 0)
```

The first two vector indices represent whether the atom is a carbon or oxygen atom, [1, 0] or [0, 1] respectively. Subsequent vector indices represent the number of single bonds, double bonds, triple bonds, aromatic bonds and disconnected bonds the atom has.

# Contributing, Reporting Issues and Other Support:

To contribute to GraphChem, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com).
