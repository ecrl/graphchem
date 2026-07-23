from rdkit import Chem

from graphchem.preprocessing.features import (
    MoleculeEncoder,
    Tokenizer,
    atom_to_str,
    bond_to_str,
    get_ring_size,
    load_encoder,
)


# Helper function to create a simple molecule for testing
def create_test_molecule(smiles):

    return Chem.MolFromSmiles(smiles)


# Test cases for get_ring_size (atom and bond paths)
def test_get_ring_size():

    # Create a benzene ring (C6H6)
    mol = create_test_molecule("C1=CC=CC=C1")

    # Check ring size for each atom in the molecule
    for atom in mol.GetAtoms():
        assert get_ring_size(atom) == 6

    # Bond path: aromatic ring bonds are size 6
    for bond in mol.GetBonds():
        assert get_ring_size(bond) == 6

    # Create an ethane molecule (C2H6)
    mol = create_test_molecule("CC")

    # Check that there are no rings in this molecule
    for atom in mol.GetAtoms():
        assert get_ring_size(atom) == 0
    for bond in mol.GetBonds():
        assert get_ring_size(bond) == 0

    # Test max_size parameter on atoms and bonds
    mol = create_test_molecule("C1=CC=CC=C1")
    for atom in mol.GetAtoms():
        assert get_ring_size(atom, max_size=5) == 5
    for bond in mol.GetBonds():
        assert get_ring_size(bond, max_size=5) == 5


# Property-level checks for atom_to_str (full strings locked by encoding oracles)
def test_atom_to_str():

    mol = create_test_molecule("C1=CC=CC=C1")
    for atom in mol.GetAtoms():
        assert atom.GetSymbol() == "C"
        assert atom.GetIsAromatic() is True
        assert atom.GetHybridization() == Chem.HybridizationType.SP2
        assert get_ring_size(atom) == 6
        token = atom_to_str(atom)
        assert isinstance(token, str) and token
        assert "'C'" in token
        assert "True" in token  # aromatic flag appears in the tuple dump
        assert token.endswith(", 6)")


# Property-level checks for bond_to_str (full strings locked by encoding oracles)
def test_bond_to_str():

    mol = create_test_molecule("C1=CC=CC=C1")
    for bond in mol.GetBonds():
        assert bond.GetBondType() == Chem.BondType.AROMATIC
        assert bond.GetIsConjugated() is True
        assert get_ring_size(bond) == 6
        token = bond_to_str(bond)
        assert isinstance(token, str) and token
        assert "AROMATIC" in token
        assert "True" in token
        assert "['C', 'C']" in token
        assert ", 6," in token


# Test cases for max_size parameter in get_ring_size
def test_get_ring_size_max_size():

    # Create a benzene ring (C1=CC=CC=C1)
    mol = create_test_molecule("C1=CC=CC=C1")

    # Check ring size with different max_size values (atoms and bonds)
    for atom in mol.GetAtoms():
        assert get_ring_size(atom, max_size=5) == 5
        assert get_ring_size(atom, max_size=6) == 6
        assert get_ring_size(atom, max_size=7) == 6
    for bond in mol.GetBonds():
        assert get_ring_size(bond, max_size=5) == 5
        assert get_ring_size(bond, max_size=6) == 6
        assert get_ring_size(bond, max_size=7) == 6


# Test cases adding tokens to a Tokenizer object
def test_tokenizer():

    tokenizer = Tokenizer()
    # Test training mode
    tokenizer.train = True
    assert tokenizer("test") == 2
    assert tokenizer("another_test") == 3

    # Test inference mode
    tokenizer.train = False
    assert tokenizer("unknown_token") == 1
    assert tokenizer.unknown == ["unknown_token"]


# Test cases for MoleculeEncoder class, vocab sizes
def test_molecule_encoder_vocab_sizes():

    smiles_list = ["C", "N"]
    encoder = MoleculeEncoder(smiles_list)
    vocab_sizes = encoder.vocab_sizes
    assert len(vocab_sizes) == 2
    # Check the number of unique atoms and bonds
    assert vocab_sizes[0] > 1
    assert vocab_sizes[1] > 1


# Test cases for MoleculeEncoder class, feature shapes
def test_molecule_encoder_encode():

    smiles = "C=C"
    encoder = MoleculeEncoder([smiles])
    enc_atoms, enc_bonds, connectivity = encoder.encode(smiles)
    # Check the shape of output tensors
    assert enc_atoms.shape == (2,)
    assert enc_bonds.shape == (2 * 1,)  # one bond, seen from two atoms
    assert connectivity.shape == (2, 2 * 1)


# Test cases for MoleculeEncoder class, encoding multiple molecules
def test_molecule_encoder_encode_many():

    smiles_list = ["C", "N"]
    encoder = MoleculeEncoder(smiles_list)
    encodings = encoder.encode_many(smiles_list)
    # Check the number of encodings
    assert len(encodings) == len(smiles_list)
    # Check the shape of each encoding
    for atoms, bonds, connectivity in encodings:
        assert len(atoms) > 0
        if len(bonds) > 0 and connectivity is not None:
            assert connectivity.shape == (2, len(bonds))


# Test cases for MoleculeEncoder class, saving and loading encoders
def test_molecule_encoder_save_load(tmpdir):

    smiles_list = ["C", "N"]
    encoder = MoleculeEncoder(smiles_list)
    filename = tmpdir.join("encoder.pkl")
    # Save the encoder
    encoder.save(str(filename))
    # Load the encoder
    loaded_encoder = load_encoder(str(filename))
    assert loaded_encoder.vocab_sizes == encoder.vocab_sizes
