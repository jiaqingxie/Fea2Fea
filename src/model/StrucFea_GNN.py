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

class StrucFeaGNN(nn.Module):
    def __init__(self, concat_fea_num = 2, embed_method = 'GIN', input_dim = 1024, output_dim = 7, depth = 2 ):
        super(StrucFeaGNN, self).__init__()
        self.concat_fea_num = concat_fea_num
        self.embed_method = embed_method
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.pre_mlp1 = nn.linear(self.concat_fea_num, 16)
        self.pre_mlp2 = nn.linear(16, 64)
        self.pre_mlp3 = nn.linear(self.input_dim - self.concat_fea_num, 256)
        self.pre_mlp4 = nn.linear(256, 64)
        self.post_mlp1 = nn.linear(128, 32)
        self.post_mlp2 = nn.linear(32, output_dim)

        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(self.depth):
            self.mlps.append(nn.Sequential(
                            nn.Linear(128, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Linear(128, 128),
        ))
        if embed_method == 'GIN':
            for i in range(self.depth):
                self.convs.append(GINConv(self.mlps[i]))
    
    def forward(self, data):
        ident_idx = self.input_dim - self.concat_fea_num

        ident_vec = data[:,:ident_idx]
        struc_vec = data[:,-self.concat_fea_num:]

        x = F.relu(self.pre_mlp1(struc_vec))
        x = F.dropout(x)
        x = F.relu(self.pre_mlp2(x))
        x = F.dropout(x)

        init_x = F.relu(self.pre_mlp3(ident_vec))
        init_x = F.dropout(init_x)
        init_x = F.relu(self.pre_mlp4(init_x))
        init_x = F.dropout(init_x)

        new_x = torch.cat((init_x, x), dim = 1)
        
        # two gnn layers
        graph_embed_0 = self.convs[0](new_x)
        graph_embed_0+=new_x # skip connection
        graph_embed_1 = self.convs[1](graph_embed_0)
        graph_embed_1 = graph_embed_1 + graph_embed_0 + new_x # skip connection

        output = F.relu(self.post_mlp1(graph_embed_1))
        output = self.post_mlp2(output)

        return F.log_softmax(output, dim =1)