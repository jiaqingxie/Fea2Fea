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
    def __init__(self, concat_fea = True, concat_fea_num = 2, embed_method = 'GIN', input_dim = 1024, output_dim = 7, depth = 2, cat_method = 'SimpleConcat', required_batch = True, embed_dim = 16):
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

        self.pre_mlp1 = nn.Linear(self.concat_fea_num, self.embed_dim)
        self.pre_mlp2 = nn.Linear(self.embed_dim, self.embed_dim * 2)
        self.pre_mlp3 = nn.Linear(self.input_dim - self.concat_fea_num, self.embed_dim)
        self.pre_mlp4 = nn.Linear(self.embed_dim, self.embed_dim * 2)
        self.pre_mlp5 = nn.Linear(self.embed_dim, self.embed_dim * 4)
        self.post_mlp1 = nn.Linear(self.embed_dim * 4, self.embed_dim * 2)
        self.post_mlp2 = nn.Linear(self.embed_dim * 2, output_dim)
        
        self.bilinear = nn.Bilinear(self.embed_dim * 2, self.embed_dim * 2,self.embed_dim * 4)
        self.ntn = NeuralTensorNetwork(self.embed_dim * 2, self.embed_dim * 4)
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.depth):
            self.mlps.append(nn.Sequential(
                            nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
                            nn.BatchNorm1d(self.embed_dim * 4),
                            nn.ReLU(),
                            nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
            ))
            self.batch_norms.append(
                            nn.BatchNorm1d(self.embed_dim * 4,)
            )
        
        
        for i in range(self.depth):
            if embed_method == 'GIN':
                self.convs.append(GINConv(self.mlps[i]))
            elif embed_method == 'GCN':
                self.convs.append(GCNConv(self.embed_dim * 4, self.embed_dim * 4, cached=False))
            elif embed_method == 'GAT':
                self.convs.append(GATConv(self.embed_dim * 4, self.embed_dim * 4, heads= 4, concat = False, dropout= 0.4))
            elif embed_method == 'SAGE':
                self.convs.append(SAGEConv(self.embed_dim * 4,self.embed_dim * 4, normalize=True))
            elif embed_method == 'None' or embed_method == 'MLP' :
                pass
            else:
                print("please give four of the embedding methods: GIN, GAT, GCN, SAGE")
                sys.exit(0)
        
    def forward(self, data):
        
        ident_idx = self.input_dim - self.concat_fea_num
        ident_vec = data.x[:,:ident_idx]
        struc_vec = data.x[:,-self.concat_fea_num:]
        
        x = F.relu(self.pre_mlp1(struc_vec)) 
        #x = F.dropout(x)
        x = F.relu(self.pre_mlp2(x))
        #x = F.dropout(x)

        init_x = F.relu(self.pre_mlp3(ident_vec))
        #init_x = F.dropout(init_x)
        if self.concat_fea:
            init_x = F.relu(self.pre_mlp4(init_x))
        else:
            init_x_ = F.relu(self.pre_mlp5(init_x))
        if self.cat_method == 'SimpleConcat':
            new_x = torch.cat((init_x, x), dim = 1) if self.concat_fea == True else init_x_
        elif self.cat_method == 'Bilinear':
            new_x = self.bilinear(init_x, x)
        elif self.cat_method == 'NTN':
            new_x = self.ntn(init_x, x)
        
        if self.embed_method != 'MLP':
            # two gnn layers
            graph_embed_0 = self.batch_norms[0](self.convs[0](new_x, data.edge_index))
            graph_embed_0 = F.dropout(graph_embed_0, p = 0.3, training=self.training)
            graph_embed_0+=new_x # skip connection
            graph_embed_1 = self.batch_norms[1](self.convs[1](graph_embed_0, data.edge_index))
            graph_embed_1 = F.dropout(graph_embed_1, p = 0.3, training=self.training)
            graph_embed_1 = graph_embed_1 + graph_embed_0 + new_x # skip connection
            # readout layer
            if self.required_batch:
                tmp = global_mean_pool(graph_embed_1, data.batch)  
            else:
                tmp = graph_embed_1
            tmp = F.dropout(tmp, p= 0.3, training=self.training)
        else:
            if self.required_batch:
                tmp = global_mean_pool(new_x, data.batch)   
            else:
                tmp = new_x     
            tmp = F.dropout(tmp, p= 0.3, training=self.training)


        output = F.relu(self.post_mlp1(tmp))
        output = self.post_mlp2(output)

        return F.log_softmax(output, dim =1)