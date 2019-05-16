from graphchem import Molecule
from csv import DictReader
import csv


def main():

    with open('mols.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            mol = Molecule(row['smiles'])
            print('Molecule SMILES: {}'.format(mol.smiles))
            print(mol)


if __name__ == '__main__':

    main()
