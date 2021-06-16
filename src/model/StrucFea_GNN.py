import itertools
import os.path as osp
from random import triangular
from numpy.lib.function_base import append
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
from torch_geometric.nn import global_mean_pool

class StrucFeaGNN(nn.Module):
    def __init__(self, concat_fea = True, concat_fea_num = 2, embed_method = 'GIN', input_dim = 1024, output_dim = 7, depth = 3):
        super(StrucFeaGNN, self).__init__()
        self.concat_fea_num = concat_fea_num
        self.embed_method = embed_method
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.concat_fea = concat_fea
        self.pre_mlp1 = nn.Linear(self.concat_fea_num, 16)
        self.pre_mlp2 = nn.Linear(16, 32)
        self.pre_mlp3 = nn.Linear(self.input_dim - self.concat_fea_num, 16)
        self.pre_mlp4 = nn.Linear(16, 32)
        self.post_mlp1 = nn.Linear(64, 16)
        self.post_mlp2 = nn.Linear(16, output_dim)

        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.depth):
            self.mlps.append(nn.Sequential(
                            nn.Linear(64, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Linear(64, 64),
            ))
            self.batch_norms.append(
                            nn.BatchNorm1d(64)
            )
        
        if embed_method == 'GIN':
            for i in range(self.depth):
                self.convs.append(GINConv(self.mlps[i]))
    
    def forward(self, data, batch):

        ident_idx = self.input_dim - self.concat_fea_num
        ident_vec = data.x[:,:ident_idx]
        struc_vec = data.x[:,-self.concat_fea_num:]

        x = F.relu(self.pre_mlp1(struc_vec))
        #x = F.dropout(x)
        x = F.relu(self.pre_mlp2(x))
        #x = F.dropout(x)

        init_x = F.relu(self.pre_mlp3(ident_vec))
        #init_x = F.dropout(init_x)
        init_x = F.relu(self.pre_mlp4(init_x))
        #init_x = F.dropout(init_x)

        new_x = torch.cat((init_x, x), dim = 1)
        
        # two gnn layers
        graph_embed_0 = self.batch_norms[0](self.convs[0](new_x, data.edge_index))
        graph_embed_0 = F.dropout(graph_embed_0, p = 0.5, training=self.training)
        graph_embed_0+=new_x # skip connection
        graph_embed_1 = self.batch_norms[1](self.convs[1](graph_embed_0, data.edge_index))
        graph_embed_1 = F.dropout(graph_embed_1, p = 0.5, training=self.training)
        graph_embed_1 = graph_embed_1 + graph_embed_0 + new_x # skip connection

        # readout layer
        tmp = global_mean_pool(graph_embed_1, data.batch)        
        tmp = F.dropout(tmp, p= 0.5, training=self.training)
        output = F.relu(self.post_mlp1(tmp))
        output = self.post_mlp2(output)

        return F.log_softmax(output, dim =1)