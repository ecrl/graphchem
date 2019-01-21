from graphchem import GraphMapper
import csv


def main():

    smiles = []
    cetane_nums = []
    with open('mols.csv', 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            smiles.append(row['SMILES'])
            cetane_nums.append(row['CN'])

    gm = GraphMapper()
    for idx, smi in enumerate(smiles):
        gm.add_graph(smi, cetane_nums[idx])
    gm.train()
    preds = gm.predict()
    for idx, pred in enumerate(preds):
        print(pred, gm._targets[idx])


if __name__ == '__main__':

    main()
