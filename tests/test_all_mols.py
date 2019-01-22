from graphchem import Graph
import csv


def main():

    with open('mols.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            g = Graph(row['smiles'])
            print('\nMolecule: {}'.format(g.smiles))
            print(g)
            for atom in g.pack():
                print(atom)


if __name__ == '__main__':

    main()
