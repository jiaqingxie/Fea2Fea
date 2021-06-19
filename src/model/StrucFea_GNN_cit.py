import itertools
import os.path as osp
import sys
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
from model.aug_GNN import NeuralTensorNetwork

class StrucFeaGNN(nn.Module):
    def __init__(self, concat_fea = True, concat_fea_num = 2, embed_method = 'GIN', input_dim = 1024, output_dim = 7, depth = 2, cat_method = 'SimpleConcat', required_batch = True, embed_dim = 8):
        super(StrucFeaGNN, self).__init__()
        self.concat_fea_num = concat_fea_num
        self.embed_method = embed_method
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.concat_fea = concat_fea
        self.cat_method = cat_method
        self.required_batch = required_batch
        self.embed_dim = embed_dim
        in_dim = 0
        if self.concat_fea:
            in_dim = self.input_dim - self.concat_fea_num + self.embed_dim * 2
        else:
            in_dim = self.input_dim - self.concat_fea_num

        self.pre_mlp = nn.ModuleList()
        self.bilinear = nn.ModuleList()
        self.ntn = nn.ModuleList()
        if self.concat_fea_num == 3:
            self.e_mlp = nn.Linear(self.embed_dim * 3, self.embed_dim * 2)
        elif self.concat_fea_num == 2:
            self.e_mlp = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
        for i in range(self.concat_fea_num):
            self.pre_mlp.append(nn.Linear(1, self.embed_dim))
        for i in range(self.concat_fea_num-1):
            if i == self.concat_fea_num-2:
                self.bilinear.append(nn.Bilinear(self.embed_dim, self.embed_dim, self.embed_dim * 2))
                self.ntn.append(NeuralTensorNetwork(self.embed_dim, self.embed_dim * 2))
            else:
                self.bilinear.append(nn.Bilinear(self.embed_dim, self.embed_dim, self.embed_dim))
                self.ntn.append(NeuralTensorNetwork(self.embed_dim, self.embed_dim))
        
        if self.embed_method != 'MLP':
            self.post_mlp1 = nn.Linear(self.embed_dim * 4, self.embed_dim * 2)
        else:
            self.post_mlp1 = nn.Linear(in_dim, self.embed_dim * 2)
        self.post_mlp2 = nn.Linear(self.embed_dim * 2, output_dim)
        
        #self.bilinear = nn.Bilinear(self.embed_dim * 2, self.embed_dim * 2,self.embed_dim * 4)
        #self.ntn = NeuralTensorNetwork(self.embed_dim * 2, self.embed_dim * 4)
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(self.depth):
            if i == 0:
                self.mlps.append(nn.Sequential(
                nn.Linear(in_dim, self.embed_dim * 4),
                nn.BatchNorm1d(self.embed_dim * 4),
                nn.ReLU(),
                nn.Linear(self.embed_dim * 4, self.embed_dim * 4),))
            else:
                self.mlps.append(nn.Sequential(
                nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
                nn.BatchNorm1d(self.embed_dim * 4),
                nn.ReLU(),
                nn.Linear(self.embed_dim * 4, self.embed_dim * 4),))
            self.batch_norms.append(
                            nn.BatchNorm1d(self.embed_dim * 4,)
            )

        for i in range(self.depth):
            if embed_method == 'GIN':
                self.convs.append(GINConv(self.mlps[i]))
            elif embed_method == 'GCN':
                if i == 0:
                    self.convs.append(GCNConv(in_dim, self.embed_dim * 4, cached=True))
                else:
                    self.convs.append(GCNConv(self.embed_dim * 4, self.embed_dim * 4, cached=True))
            elif embed_method == 'GAT':
                if i == 0:
                    self.convs.append(GATConv(in_dim, self.embed_dim * 4, heads= 4, concat = False, dropout= 0.4))
                else:
                    self.convs.append(GATConv(self.embed_dim * 4, self.embed_dim * 4, heads= 4, concat = False, dropout= 0.4))
            elif embed_method == 'SAGE':
                if i == 0:
                    self.convs.append(SAGEConv(in_dim, self.embed_dim * 4, normalize=True))
                else:
                    self.convs.append(SAGEConv(self.embed_dim * 4,self.embed_dim * 4, normalize=True))
            elif embed_method == 'None' or embed_method == 'MLP' :
                pass
            else:
                print("please give four of the embedding methods: GIN, GAT, GCN, SAGE")
                sys.exit(0)
        
    def forward(self, data):
        
        ident_idx = self.input_dim - self.concat_fea_num
        
        ident_vec = data.x[:,:ident_idx]
        #print(ident_vec.shape)
        struc_vec = data.x[:,-self.concat_fea_num:]
       # print(struc_vec.shape)
        _x = []
        for i in range(self.concat_fea_num):
            _x.append(F.relu(self.pre_mlp[i](struc_vec[:, i:i+1]))) # two or three N * self.embed_dim

        for i in range(self.concat_fea_num-1):
            if self.cat_method == 'SimpleConcat':
                if i == 0:
                    x = torch.cat((_x[i], _x[i+1]), dim = 1)
                else:
                    x = torch.cat((x, _x[i + 1]), dim = 1)
                if  i == self.concat_fea_num-2:
                    x = self.e_mlp(x)
       
            elif self.cat_method == 'Bilinear':
                if i == 0:
                    x = self.bilinear[i](_x[i], _x[i+1])
                else:
                    x = self.bilinear[i](x, _x[i+1])
            elif self.cat_method == 'NTN':
                if i == 0:
                    x = self.ntn[i](_x[i], _x[i+1])
                else:
                    x = self.ntn[i](x, _x[i+1])

        init_x = ident_vec
        #print(init_x.shape)
        new_x = torch.cat((init_x, x), dim = 1) if self.concat_fea == True else init_x
        #print(new_x.shape)
        if self.embed_method != 'MLP':
            # two gnn layers
            graph_embed_0 = self.convs[0](new_x, data.edge_index)
            graph_embed_0 = F.relu(graph_embed_0)
            graph_embed_0 = F.dropout(graph_embed_0, training=self.training)
            #graph_embed_0+=new_x # skip connection
            graph_embed_1 = self.convs[1](graph_embed_0, data.edge_index)
            graph_embed_1 = F.relu(graph_embed_1)
            graph_embed_1 = F.dropout(graph_embed_1, p = 0.6, training=self.training)
            graph_embed_1 = graph_embed_1 + graph_embed_0# skip connection
            graph_embed_2 = self.convs[2](graph_embed_1, data.edge_index)
            #graph_embed_2 = F.relu(graph_embed_2)
            graph_embed_2 = F.dropout(graph_embed_2, p = 0.6, training=self.training)
            graph_embed_2 = graph_embed_2 + graph_embed_1 + graph_embed_0# skip connection
            # readout layer
            if self.required_batch:
                tmp = global_mean_pool(graph_embed_2, data.batch)
                tmp = F.dropout(tmp, p= 0.5, training=self.training)
            else:
                tmp = graph_embed_2
        else:
            if self.required_batch:
                tmp = global_mean_pool(new_x, data.batch) 
                tmp = F.dropout(tmp, p= 0.5, training=self.training) 
            else:
                tmp = new_x   
                
        output = F.relu(self.post_mlp1(tmp))
        output = self.post_mlp2(output)

        return F.log_softmax(output, dim =1)