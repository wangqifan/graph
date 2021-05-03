import pandas as pd
from sklearn.model_selection import train_test_split
import torch_geometric.data as gdata
import torch
from rdkit import Chem
from collections import defaultdict
import numpy as np
from util import gen_features
import numpy
from GCN import GCN, Net
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

filename = "BBBP.csv"
smiles_tasks_df = pd.read_csv(filename)
smiles_tasks_df["smiles"] = smiles_tasks_df["smiles"].astype(str)
smilesList = smiles_tasks_df.smiles.values
print("number of all smiles: ",len(smilesList))
device = torch.device('cuda')

dataset = []

for index, row in smiles_tasks_df.iterrows():
    smiles = row["smiles"]
    property = row["BBBP"]
    mdata = gen_features(smiles, property)
    if mdata is None:
        continue
    dataset.append(mdata)
    print(mdata)
import random

random.shuffle(dataset)

print(dataset[0])

train_dataset = dataset[:1230]
test_dataset = dataset[1230:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')


from torch_geometric.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(type(train_loader))
print(type(train_dataset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(hidden_channels=64,input_features = 2, output_classes=2)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    total_loss = 0.
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.max(dim=1)[1]
        print(pred)
        total_correct += pred.eq(data.y).sum().item()
    return total_correct / len(loader.dataset)


for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')