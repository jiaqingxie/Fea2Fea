import itertools
import os.path as osp
import pandas as pd
import numpy as np
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import ARMAConv,AGNNConv
from torch_geometric.nn import GINConv,GATConv,GCNConv
from torch_geometric.nn import SAGEConv,SplineConv
import math
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.cm as cm
from torch.optim.lr_scheduler import StepLR
sys.path.append('/home/jiaqing/桌面/Fea2Fea/property_process/')
from graph_property import G_property, binning
from model.GNN import Net, debug_MLP

def train(data, train_idx):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[train_idx], data.y[train_idx]).backward()
    optimizer.step()

def valid(data, valid_idx):
    model.eval()
    with torch.no_grad():
            pred = model(data).max(dim=1)[1]
    correct = 0
    correct += pred[valid_idx].eq(data.y[valid_idx]).sum().item()
    valid_acc = correct / len(valid_idx)
    return valid_acc

def test(data, test_idx):
    model.eval()
    with torch.no_grad():
        pred = model(data).max(dim=1)[1]
    correct = 0
    correct += pred[test_idx].eq(data.y[test_idx]).sum().item()
    test_acc = correct / len(test_idx)
    return test_acc
#'MLP',
#------------------ Start our algorithm1 ---------------------#

if __name__ == '__main__':

    xlabels = ['Constant','Degree','Clustering_Coefficient','PageRank','Aver_Path_Len']
    num_of_nodes = [50, 200, 400, 800]
    for non in num_of_nodes:
        num_nodes = non
        #print(num_nodes)
        num_train_nodes = int(num_nodes * 0.8)
        num_valid_nodes = int(num_nodes * 0.1)
        num_test_nodes = int(num_nodes * 0.1)
        perm = torch.randperm(num_nodes)
        train_idx = perm[:num_train_nodes]
        valid_idx = perm[num_train_nodes:(num_train_nodes+num_valid_nodes)]
        test_idx = perm[(num_train_nodes+num_valid_nodes):(num_train_nodes+num_valid_nodes+num_test_nodes)]
        file_path = '/home/jiaqing/桌面/Fea2Fea/data/syn_data/'
        property_file = pd.read_csv(file_path+'geometric_graph_{}_property.txt'.format(non), sep = '\t')
        Aver = np.zeros((5,5))
        edge_idx_file = pd.read_csv(file_path+'geometric_graph_{}_edge_idx.txt'.format(non), sep = ',',header = None)

        x = torch.tensor(np.array(property_file), dtype=torch.float)
        edge_idx = torch.tensor(np.array(edge_idx_file), dtype=torch.long)
        

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        best_val_acc = test_acc = 0
        t = 0
        record_acc = 1
        total_epoch = 0 # for drawing curves
        avg_num = 10 # for iteration to count average accuracy
        
        for avg in range(avg_num):
            R = np.zeros((5,5)) # initialize our feature relationship matrix 
            R[0][0] = 1.000
            for i in range(0, 5):
                x_train = x[:,i].reshape((len(x),1))
                for j in range(1,5):
                    tmp = np.array(x[:,j])
                    y = binning(tmp, k = 2,data_len = len(x))
                    data = Data(x=x_train, edge_index=edge_idx.t().contiguous(), y =y).to(device)
                   
                    model =  Net(embedding='GIN').to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
                    #scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
                    data =  data.to(device)
                    t = 0
                    best_val_acc = test_acc = 0 
                    for epoch in range(1, 2000):
                        
                        train(data = data,train_idx = train_idx)
                        val_acc, tmp_test_acc = valid(data=data,valid_idx=valid_idx), test(data=data,test_idx=test_idx)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            test_acc = tmp_test_acc
                            if i == 0:
                                R[i][j] = round(test_acc,3)
                                R[j][i] = round(test_acc,3)
                            else:
                                R[i][j] = round(test_acc,3)
                            t = 0
                        t = t + 1
                        if t > 500:
                            break   
                        
                        log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'
                        #print(log.format(epoch, best_val_acc, test_acc))
                        # do that for several times
                        #scheduler.step()
                    if i == 4 and j == 4:
                        Aver = Aver + R
                        
            if avg == 0:
                k = Aver / 1
                np.set_printoptions(precision=3)
                k = pd.DataFrame(k)
                filepath = '/home/jiaqing/桌面/Fea2Fea/Result/syn_data'
                fig_name = '/{}_GIN.txt'.format(non)
                fig_path = filepath + fig_name
                k.to_csv(fig_path, header = None, index = None, sep = '\t')
                #----------- save Heatmap Matrix-----------#
                filepath = '/home/jiaqing/桌面/Fea2Fea/Result/syn_data'
                fig_name = '/{}_GIN_property.eps'.format(non)
                fig_path = filepath + fig_name
                xlabels = ['Constant','Degree','Clustering','PageRank','Aver_Path_Len']
                ylabels = ['Constant','Degree','Clustering','PageRank','Aver_Path_Len']
                cm = sns.heatmap(k,annot=True,cmap="Blues",cbar = False, square=True,
                            xticklabels = xlabels, yticklabels = ylabels)
                cm.set_xticklabels(cm.get_xticklabels(), rotation=30)
                cm.set_yticklabels(cm.get_xticklabels(), rotation=0)
                label = 'GIN'
                cm.set_title(label)
                heatmap = cm.get_figure()
                heatmap.savefig(fig_path, dpi = 800,bbox_inches='tight')
                break
            