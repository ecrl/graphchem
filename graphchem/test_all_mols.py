import csv
import graph


def main():

    with open('mols.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            g = graph.Graph(row['smiles'])
            p = g.pack()
            print()
            print(g.smiles)
            print()
            print(g)
            for i in p:
                print(i)


if __name__ == '__main__':

    main()
