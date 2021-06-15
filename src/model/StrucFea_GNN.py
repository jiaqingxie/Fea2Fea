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
    def __init__(self, concat_fea_num = 2, embed_method = 'GIN', dim = 1024):
        super(StrucFeaGNN, self).__init__()
        self.concat_fea_num = concat_fea_num
        self.dim = dim
        self.pre_mlp1 = nn.linear(self.concat_fea_num, 16)
        self.pre_mlp2 = nn.linear(16, 64)
        self.pre_mlp3 = nn.linear(self.dim - self.concat_fea_num, 256)
        self.pre_mlp4 = nn.linear(256, 64)


    
    def forward(self, data):
        ident_idx = sel.dim - self.concat_fea_num

        ident_vec = data[,:ident_idx]
        struc_vec = data[,-self.concat_fea_num:]

        x = F.relu(self.pre_mlp1(struc_vec))
        x = F.dropout(x)
        x = F.relu(self.pre_mlp2(x))
        x = F.drouput(x)

        init_x = F.relu(self.pre_mlp3(ident_vec))
        init_x = F.dropout(init_x)
        init_x = F.relu(self.pre_mlp4(init_x))
        init_x = F.drouput(init_x)

        new_x = 



        return