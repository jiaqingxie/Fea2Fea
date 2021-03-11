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
import math
import matplotlib.pyplot as plt
from graph_property import G_property, binning
import seaborn as sns

from model.GNN import Net

path = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

def test():
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_val_acc = test_acc = 0
t = 0
record_acc = 1

train_accu_plot = []
epoch_plot = []

#------------------ Start our algorithm1 ---------------------#



for dataset,embedding_method in list(itertools.product(['Cora','PubMed','Citeseer'],['SAGE','GAT','GCN','GIN'])):
    R = [[0 for i in range(5)] for j in range(5)] # initialize our feature relationship matrix 
    dataset_name = dataset
    dataset = Planetoid(path, name = dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    name = r'/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/' + dataset_name + '_property.txt'
    property_file = pd.read_csv(name, sep = '\t')
    R[0][0] = 1.000

    total_epoch = 0 # for drawing curves
    training_error = np.array([]) # for drawing curves
    
    if record_acc:
        for i in range(5):

            propert_i = property_file.iloc[:,[i]]
            array = np.array(propert_i)
            data.x = torch.tensor(array).float()
            
            for j in range(1,5):

                propert_j = property_file.iloc[:,[j]]
                array_2 = np.array(propert_j)
                number = len(data.y)
                data.y = binning(array_2, k = 6,data_len =  number)
                #print(data.y)
                model =  Net(embedding=embedding_method).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.04, weight_decay=5e-4)
                data =  data.to(device)
                t = 0
                best_val_acc = test_acc = 0 
                for epoch in range(1, 3000):
                    
                    train()
                    train_acc, val_acc, tmp_test_acc = test()
                    training_error = np.append(training_error,round(train_acc,3))
                    #train_accu_plot.append(train_acc)
                    #epoch_plot.append(epoch)
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
                    
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    print(log.format(epoch, train_acc, best_val_acc, test_acc))
                    if i == 4 and j == 4:
                        #print(R)
                        k = np.array(R)
                        k = pd.DataFrame(k)
                        filepath = '/home/jiaqing/桌面/Fea2Fea/Result/Planetoid'
                        fig_name = '/' + dataset_name + '_' + embedding_method + '.txt'
                        fig_path = filepath + fig_name
                        k.to_csv(fig_path, header = None, index = None, sep = '\t')
                        #----------- save Heatmap Matrix-----------#
                        filepath = '/home/jiaqing/桌面/Fea2Fea/Result/Planetoid'
                        fig_name = '/' + dataset_name + '_' + embedding_method + '_property' + '.eps'
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
                '''
                plt.plot(epoch_plot,train_accu_plot)
                plt.show()
                
                nb_classes = 6
                confusion_matrix = torch.zeros(nb_classes,nb_classes)
                
                logits, accs = model(), []
                for _, mask in data('test_mask'):
                    pred = logits[mask].max(1)[1]
                    #print(pred)
                    #print(data.y[mask])
                    for i in range(len(pred)):
                        confusion_matrix[pred[i]][data.y[mask][i]] = confusion_matrix[pred[i]][data.y[mask][i]]+1
                print(confusion_matrix)#
                
                with torch.no_grad():
                    print(model.embedding)
                    print(model.embedding.shape)
                '''
    




