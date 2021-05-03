import pandas as pd
from sklearn.model_selection import train_test_split
import torch_geometric.data as gdata
import torch
from rdkit import Chem
from collections import defaultdict
import numpy as np
from util import gen_features
import numpy
import torch.nn.functional as F
from GCN import AttentiveFP
from sklearn.metrics  import roc_auc_score
from torch import nn
from  torch.nn import TripletMarginLoss

filename = "HIV.csv"
smiles_tasks_df = pd.read_csv(filename)
smiles_tasks_df["smiles"] = smiles_tasks_df["smiles"].astype(str)
smilesList = smiles_tasks_df.smiles.values
print("number of all smiles: ",len(smilesList))
device = torch.device('cuda')

dataset = []

for index, row in smiles_tasks_df.iterrows():
    smiles = row["smiles"]
    property = row["HIV_active"]
    mdata = gen_features(smiles, property)
    dataset.append(mdata)
    if index % 1000 == 0:
        print(mdata)
import random
random.shuffle(dataset)

print(dataset[0])

train_dataset = dataset[:25000]
test_dataset = dataset[25000:]

from torch_geometric.data import DataLoader

def getTripe(postive, neigative, k):
    anch = DataLoader(random.sample(postive, k), batch_size = k,  shuffle=True)
    postive = DataLoader(random.sample(postive, k), batch_size = k,  shuffle=True)
    neigative =  DataLoader(random.sample(neigative, k), batch_size = k,  shuffle=True)
    return anch, postive, neigative, k

def getData(dataset, k):
    postive = []
    neigative = []
    for item in dataset:
        if item.y == 0:
            postive.append(item)
        else:
            neigative.append(item)
    if len(neigative) < k:
        k = len(neigative)

    if random.random() < 0.4:
        return getTripe(postive, neigative, k)
    else:
        return getTripe(neigative, postive, k)

    

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentiveFP(2, 2, 2, 2, 4, 0)
model = model.to(device)
#model = torch.load("model.pkl")
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

weight=torch.from_numpy(np.array([0.05,0.95])).float().cuda()
loss_c = nn.CrossEntropyLoss(weight)
loss_f = TripletMarginLoss()

def train():
    model.train()
    anch, postive, neigative, k = getData(train_dataset, 256) 
    def computer(DataLoader):
        for data in DataLoader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr ,data.batch)
            return out
    out1 = computer(anch)
    out2 = computer(postive)
    out3 = computer(neigative)

    loss = loss_f(out1, out2, out3)
    loss.backward()
    optimizer.step()

    total_loss = 0.
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr ,data.batch)
        loss = loss_c(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)

minloss = 10000

@torch.no_grad()
def test(loader):
    model.eval()
    global minloss
    total_correct = 0
    loss = 0
    testval = []
    labels = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss += loss_c(out, data.y)
        for item in out:
          val = item.cpu()
          testval.append(1 - (val[0] - val[1]))
        for item in data.y:
          labels.append(item.cpu()) 
        if loss < minloss:
            minloss = loss
            torch.save(model, 'model2.pkl')
    print(roc_auc_score(labels,testval))
    return loss


for epoch in range(1,10010):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')
