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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from chem.pretrain_deepgraphinfomax import Infomax, Discriminator
from chem.model import GNN

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
    if index > 10:
        break
import random

random.shuffle(dataset)

print(dataset[0])

train_dataset = dataset[:1230]
test_dataset = dataset[1230:]

from torch_geometric.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(type(train_loader))
print(type(train_dataset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args ={"num_layer":5,"emb_dim":300,"JK":"last","dropout_ratio":0,"gnn_type":"gin"}


#set up model
gnn = GNN(5, 300, "last", 0, "gin")
discriminator = Discriminator(300)
model = Infomax(gnn, discriminator)
model.gnn.load_state_dict(torch.load('infomax.pth'))
model = model.to(device)
@torch.no_grad()
def test(loader):
   
    total_correct = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_emb = model.gnn(batch.x, batch.edge_index, batch.edge_attr)
        print(node_emb)
       


for epoch in range(1, 101):
    test(train_loader)
    test(test_loader)