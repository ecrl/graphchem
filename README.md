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
```
$ pip install graphchem
```

### Method 2: From Source
```
$ git clone https://github.com/tjkessler/graphchem
$ cd graphchem
$ python setup.py install
```

There are currently no dependencies for GraphChem.

# Usage:

Import the Molecule object and initialize it using a SMILES string. For this example, we will use 2,5-dimethylfuran:

```python
from graphchem import Molecule
mol = Molecule('CC1=CC=C(C)O1')
```

Note: only molecules containing Boron, Carbon, Nitrogen, Oxygen, Phosphorus, Sulfur, Flourine and/or Iodine are currently supported.

If we print the Molecule, we can display each atom and its bonds:

```python
>>> print(mol)
ID      Atom    Connections
0       C       [(1, '-')]
1       C       [(0, '-'), (2, '='), (6, '-')]
2       C       [(1, '='), (3, '-')]
3       C       [(2, '-'), (4, '=')]
4       C       [(3, '='), (5, '-'), (6, '-')]
5       C       [(4, '-')]
6       O       [(4, '-'), (1, '-')]
```

Tuples in each atom's list of connections represent the ID of an atom it is bonded to and the bond type (e.g. '-' for single bond, '=' for double bond, '#' for triple bond).

Atom states are defined by a 20 item vector, containing a one-hot representation of the atom type (e.g. [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] for Boron, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] for carbon), sums of all bond types (e.g. [0, 1, 0, 0, 0, 0, 0, 0] for single bond, [0, 0, 1, 0, 0, 0, 0, 0] for double bond), and whether the atom exists in a ring or not ([1, 0] or [0, 1], respectively).

To get the states of each atom:

```python
>>> for atom in mol._atoms:
>>>     print(atom.state)
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0]
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0]
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0]
```

The graph representation of a molecule is defined as the sum of all atom states. To get the graph representation of a molecule:

```python
>>> print(mol.repr)
[0, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 4, 0, 0, 0, 0, 0, 5, 2]
```

When manipulating a graph, new node states are updated with a transition function. New states are derived from the current state of the node and the states of the node's neighbors. In GraphChem, atoms of a molecule correlate to the nodes in a graph, and can be updated with the molecule's "transition" method.

The "transition" method accepts one argument, a callable transition function. The supplied transition function should accept two arguments, a list of length 20 correpsonding to the current state of an atom, and another list of length 20 corresponding to the sum of the states of the atom's neighbors. The supplied transition function should return a list of length 20, the new state for a given atom:

```python
def average_of_atom_and_neighbors(atom_state, neighbor_states):
    '''Example transition function, averaging indices of the atom's state and
    its neighbors' states

    Args:
        atom_state (list): current state of the atom, list length 20
        neighbor_states (list): current (summed) states of the atom's
            neighbors, list length 20

    Returns:
        list: new state of the atom, list length 20
    '''
    new_state = []
    for i in range(len(atom_state)):
        new_state.append((atom_state[i] + neighbor_states[i]) / 2)
    return new_state

mol = Molecule('CC1=CC=C(C)O1')
mol.transition(average_of_atom_and_neighbors)
```

The new atom states and graph representation of the molecule are:

```python
>>> for atom in mol._atoms:
>>>     print(atom.state)
[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
[0.0, 1.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.5]
[0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0]
[0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0]
[0.0, 1.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.5]
[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
[0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0]

>>> print(mol.repr)
[0.0, 9.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.5, 2.0]
```

# Contributing, Reporting Issues and Other Support:

To contribute to GraphChem, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com).
