import itertools
import os.path as osp
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import ARMAConv,AGNNConv
from torch_geometric.nn import GINConv,GATConv,GCNConv
from torch_geometric.nn import SAGEConv,SplineConv

class Net(nn.Module):
    def __init__(self, embedding):
        super(Net, self).__init__()
        mlp1 = nn.Sequential(
                nn.Linear(1, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512,256),
            )
        mlp2 = nn.Sequential(
                nn.Linear(256,128 ),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128,64),
            )
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.embedding = embedding

        if self.embedding == 'SAGE':
            self.conv1 = SAGEConv(1,256,normalize=True)
            self.conv2 = SAGEConv(256,64 ,normalize=True)
        elif self.embedding == 'GAT':
            self.conv1 = GATConv(1, 16,heads= 16, dropout=0.6)
            self.conv2 = GATConv(16 * 16, 64, heads=1, concat=False,
                           dropout=0.6)
        elif self.embedding == 'GCN':
            self.conv1 = GCNConv(1,256,cached=False)
            self.conv2 = GCNConv(256,64,cached=False)
        elif self.embedding == 'GIN':
            self.conv1 = GINConv(mlp1)
            self.conv2 = GINConv(mlp2)
        else:
            pass 
        self.lin1 = nn.Linear(64,16)
        self.lin2 = nn.Linear(16,6)
        self.latent = 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(2):
            if i == 0:
                x = self.conv1(x, edge_index, data.edge_attr)
                if self.embedding != 'GIN':
                    x = self.batch_norm1(x)
                    x = F.relu(x)
                x = F.dropout(x, training=self.training)
            if i == 1:
                x = self.conv2(x, edge_index, data.edge_attr)
                if self.embedding != 'GIN':
                    x = self.batch_norm2(x)
                    x = F.relu(x)
                x = F.dropout(x, training=self.training)
                graph_embedding = F.dropout(x, training=self.training)
                self.latent = graph_embedding
        x = F.relu(self.lin1(self.latent))
        x = self.lin2(x)
        return F.log_softmax(x, dim =1)