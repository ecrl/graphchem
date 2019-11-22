from graphchem import GNN, Molecule


def test_train_atom_count():

    smiles = [
        'CCC',
        'CCCCC',
        'CCCCCCC'
    ]
    mols = [Molecule(s) for s in smiles]
    targets = [3, 5, 7]

    gnn = GNN()
    gnn.train(mols, targets, verbose=True)

    new_smiles = [
        'CC',
        'CCCC',
        'CCCCCC',
        'CCCCCCCC'
    ]
    new_mols = [Molecule(s) for s in new_smiles]
    preds = gnn.predict(new_mols)
    print(preds)
    print([2, 4, 6, 8])


if __name__ == '__main__':

    test_train_atom_count()
