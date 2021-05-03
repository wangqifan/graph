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
from torch_geometric.nn.models.attentive_fp import AttentiveFP
from sklearn.metrics  import roc_auc_score

filename = "HIV.csv"
smiles_tasks_df = pd.read_csv(filename)


print(smiles_tasks_df.HIV_active.value_counts())