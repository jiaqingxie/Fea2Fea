import itertools
import os.path as osp
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn import ARMAConv,AGNNConv,global_mean_pool
from torch_geometric.nn import GINConv,GATConv,GCNConv
from torch_geometric.nn import SAGEConv,SplineConv
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import DataLoader, Dataset
from torch_scatter import scatter_mean
from graph_property import G_property,binning
#import torch_xla.core.xla_model as xm
from model.GNN import Net

def reserve(task, dn,  loader):
    t = 0
    for load in loader:
        G = []
        # construct graph
        for p1 in range(np.array(load.edge_index).shape[1]):
            G.append((int(load.edge_index[0][p1]),int(load.edge_index[1][p1])))
        # calculate graph properties
        constant = G_property(G, constant_bool=1)
        degrees, graph = G_property(G, degree_bool=1, bin_bool=0) 
        clustering, graph = G_property(G, clustering_bool=1, bin_bool=0) 
        pagerank, graph = G_property(G, pagerank_bool=1, bin_bool=0)
        avg_path_len_G, graph = G_property(G, avg_path_length_bool=1, bin_bool=0)

        matrix = torch.cat((constant,degrees),1)
        matrix = torch.cat((matrix,clustering),1)
        matrix = torch.cat((matrix,pagerank),1)
        matrix = torch.cat((matrix,avg_path_len_G),1)

        matrix = matrix.numpy()
        matrix = pd.DataFrame(matrix,columns = ['Constant_feature','Degree','Clustering_coefficient','Pagerank','Aver_path_len'])   
        name = r'/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/' + dn + '/' + dn + '_property' + str(t) + task +'.txt'
        matrix.to_csv(name, sep = '\t', index=False)
        t+=1

def train(i, j, dn, model, task, optimizer, train_loader, device, k = 6):
    total_loss = 0
    model.train()
    total_num_nodes = 0
    t= 0
    for load in train_loader:
        name = r'/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/' + dn + '/' + dn + '_property' + str(t) + task +'.txt'
        property_file = pd.read_csv(name, sep = '\t')

        propert_i = property_file.iloc[:,[i]]
        array = np.array(propert_i)
        load.x = torch.tensor(array).float()
        #print(load.x.shape)

        propert_j = property_file.iloc[:,[j]]
        array_2 = np.array(propert_j)
        number = len(array_2)
        load.y = binning(array_2, k = k, data_len =  number)
        # --------- training loop ---------- #
        
        load = load.to(device)
        optimizer.zero_grad()
        out = model(load)
        loss = F.nll_loss(out,load.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(load.y)
        total_num_nodes+=len(load.y)
        t+=1
        #print(loss)
    train_loss = total_loss / total_num_nodes
    return train_loss

def valid(i, j, dn, model, task, optimizer, valid_loader, device, k = 6):
    correct = 0
    model.eval()
    total_num_nodes = 0
    t = 0
    for load in valid_loader:
        name = r'/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/' + dn + '/' + dn + '_property' + str(t) + task +'.txt'
        property_file = pd.read_csv(name, sep = '\t')

        propert_i = property_file.iloc[:,[i]]
        array = np.array(propert_i)
        load.x = torch.tensor(array).float()


        propert_j = property_file.iloc[:,[j]]
        array_2 = np.array(propert_j)
        number = len(array_2)
        load.y = binning(array_2, k = k,data_len =  number)
        
        with torch.no_grad():
            load = load.to(device)
            pred = model(load).max(dim=1)[1]
        correct += pred.eq(load.y).sum().item()
        total_num_nodes+=len(load.y)
        t+=1
    valid_acc = correct / total_num_nodes
    return valid_acc

def test(i, j, dn,  model, task, optimizer, test_loader, device, k = 6):
    correct = 0
    model.eval()
    total_num_nodes = 0
    t = 0
    for load in test_loader:

        name = r'/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/' + dn + '/' + dn + '_property' + str(t) + task +'.txt'
        property_file = pd.read_csv(name, sep = '\t')

        propert_i = property_file.iloc[:,[i]]
        array = np.array(propert_i)
        load.x = torch.tensor(array).float()


        propert_j = property_file.iloc[:,[j]]
        array_2 = np.array(propert_j)
        number = len(array_2)
        load.y = binning(array_2, k = k,data_len =  number)
        
        with torch.no_grad():
            load = load.to(device)
            pred = model(load).max(dim=1)[1]
        correct += pred.eq(load.y).sum().item()
        total_num_nodes+=len(load.y)
        t+=1
    test_acc = correct / total_num_nodes
    return test_acc

if __name__ == '__main__':

    dataset_name = ['ENZYMES','PROTEINS', 'NCI1']

    GNN_model = ['GIN','SAGE','GAT', 'GCN']
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    for dn, embedding_method in list(itertools.product(dataset_name, GNN_model)):
        # make sure that your dataset is reserved in /tmp/dn/dn/...
        dataset = TUDataset(root = '/home/jiaqing/桌面/Fea2Fea/data/' + dn, name = dn, use_node_attr = False)
        # batch size is the parameter
        # print(len(dataset))
        train_len, valid_len= int(0.8 * len(dataset)), int(0.1 * len(dataset))
        test_len = len(dataset) - train_len - valid_len
        train_loader = DataLoader(dataset[0:train_len], batch_size = 16, shuffle=False)
        valid_loader = DataLoader(dataset[train_len:(train_len+valid_len)], batch_size = 16, shuffle = False)
        test_loader = DataLoader(dataset[(train_len+valid_len):len(dataset)], batch_size = 16, shuffle = False)
        # for each batch, calculate the feature properties
        #reserve('train', dn, train_loader)
        #reserve('valid', dn, valid_loader)
        #reserve('test', dn, test_loader)

        R = [[0 for i in range(5)] for j in range(5)] # initialize our feature relationship matrix 
        R[0][0] = 1.000
        # i is the featire taken as input,  j is the predicted feature
        for i in range(5):
            for j in range(1,5):
                model = Net(embedding=embedding_method).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr = 0.04)
                #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
                # record epoch
                best_epoch = 0
                best_valid_acc = 0
                best_test_acc = 0
                op_iters = 0
                for epoch in range(1, 200):
                    if dn == 'NCI1':
                        if j == 2 or i == 2:
                            R[i][j] = 0
                            R[j][i] = 0
                            break
                    # for train
                    t_loss = train(i, j, dn, model, 'train', optimizer, train_loader, device)
                    # for valid 
                    v_acc = valid(i, j, dn, model, 'valid', optimizer, valid_loader, device)
                    # for test
                    t_acc = test(i, j, dn, model, 'test', optimizer, test_loader, device)
                    print('Epoch {:03d}, Train Loss: {:.4f}, Valid acc :{:.4f}, Test acc : {:.4f}'.format(
                        epoch, t_loss, v_acc, t_acc ))

                    if v_acc > best_valid_acc:
                        best_valid_acc = v_acc
                        best_test_acc = t_acc
                        best_epoch = epoch
                        
                        if i == 0:
                            R[i][j] = round(t_acc,3)
                            R[j][i] = round(t_acc,3)
                        else:
                            R[i][j] = round(t_acc,3)
                        op_iters=0
                    op_iters+=1
                    if op_iters > 20:
                        break
                    if i == 4 and j == 4:
                        #print(R)
                        k = np.array(R)
                        k = pd.DataFrame(k)
                        filepath = '/home/jiaqing/桌面/Fea2Fea/Result/TUdataset'
                        fig_name = '/' +dn + '_' + embedding_method + '.txt'
                        fig_path = filepath + fig_name
                        k.to_csv(fig_path, header = None, index = None, sep = '\t')
                        #----------- save Heatmap Matrix-----------#
                        filepath = '/home/jiaqing/桌面/Fea2Fea/Result/TUdataset'
                        fig_name = '/' +dn + '_' + embedding_method + '_property' + '.eps'
                        fig_path = filepath + fig_name
                        xlabels = ['Constant','Degree','Clustering','PageRank','Aver_Path_Len']
                        ylabels = ['Constant','Degree','Clustering','PageRank','Aver_Path_Len']
                        cm = sns.heatmap(R,annot=True,cmap="Blues",cbar = False, square=True,
                                    xticklabels = xlabels, yticklabels = ylabels)
                        cm.set_xticklabels(cm.get_xticklabels(), rotation=30)
                        cm.set_yticklabels(cm.get_xticklabels(), rotation=0)
                        label = embedding_method
                        cm.set_title(label)
                        heatmap = cm.get_figure()
                        heatmap.savefig(fig_path, dpi = 400,bbox_inches='tight')
                        plt.show()
                        break
                    print('Current optimal valid_acc {:.4f} at epoch {} with test acc {:.4f}'.format(best_valid_acc , best_epoch, best_test_acc))

                    #scheduler.step()
                

