import itertools
import os.path as osp
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn import ARMAConv,AGNNConv,global_mean_pool
from torch_geometric.nn import GINConv,GATConv,GCNConv
from torch_geometric.nn import SAGEConv,SplineConv
import math
import matplotlib.pyplot as plt
#from graph_property import G_property, binning
import seaborn as sns
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean




#dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=False)
#loader = DataLoader(dataset, batch_size=1, shuffle=True)
#dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS', use_node_attr=False)
#train_loader = DataLoader(dataset[:480] , batch_size=1, shuffle=True)
dataset = TUDataset(root='/tmp/NCI1', name='NCI1', use_node_attr=False)
print(len(dataset))
loader = DataLoader(dataset, batch_size=2, shuffle=True)
#valid_loader = DataLoader(dataset[480:540], batch_size=1, shuffle=True)
i = 0
for load in loader:
    print(load.edge_index)
    break
#for data in loader:
    #print(data.x.edge_index)

'''
test_loader = DataLoader(dataset[540:600], batch_size=32, shuffle=True) 

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(Net, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25), 
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return GCNConv(input_dim, hidden_dim)
        else:
            return GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = global_mean_pool(x, batch)

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(max(dataset.num_node_features, 1), 32, dataset.num_classes, task='graph').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def train():
    total_loss = 0
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out,data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test():
    model.eval()
    correct = 0
    for data in test_loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            pred = pred.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_loader.dataset)

def valid():
    model.eval()

    correct = 0
    for data in valid_loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(valid_loader.dataset)

for epoch in range(1, 201):
    loss = train()
    valid_acc = valid()
    test_acc = test()
    print('Epoch {:03d}, Loss: {:.4f}, Valid :{:.4f},  Test: {:.4f}'.format(
        epoch, loss, valid_acc, test_acc))
    scheduler.step()
'''