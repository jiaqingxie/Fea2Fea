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

path = osp.join('/home/jiaqing/桌面/FASG_KDD/data/')

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
        self.embedding = embedding
        if self.embedding == 'SAGE':
            self.conv1 = SAGEConv(1,128,normalize=True)
            self.conv2 = SAGEConv(128,64 ,normalize=True)
        elif self.embedding == 'GAT':
            self.conv1 = GATConv(1, 16,heads= 16, dropout=0.6)
            self.conv2 = GATConv(16 * 16, 64, heads=1, concat=False,
                           dropout=0.6)
        elif self.embedding == 'GCN':
            self.conv1 = GCNConv(1,256,cached=True)
            self.conv2 = GCNConv(256,64,cached=True)
        elif self.embedding == 'GIN':
            self.conv1 = GINConv(mlp1)
            self.conv2 = GINConv(mlp2)
        else:
            pass 
        self.lin1 = nn.Linear(64,16)
        self.lin2 = nn.Linear(16,6)
        self.latent = 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, data.edge_attr))
        #x = F.relu(self.conv1(x))
        x = F.dropout(x, training=self.training)
        #x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x, edge_index, data.edge_attr))
        graph_embedding = F.dropout(x, training=self.training)
        self.latent = graph_embedding
        x = F.relu(self.lin1(graph_embedding))
        x = self.lin2(x)
        return F.log_softmax(x, dim =1)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model =  Net().to(device)


best_val_acc = test_acc = 0
t = 0
record_acc = 0


train_accu_plot = []
epoch_plot = []

#------------------ Start our algorithm1 ---------------------#



for dataset,embedding_method in list(itertools.product(['Cora','PubMed','Citeseer'],['SAGE','GAT','GCN','GIN'])):
    R = [[0 for i in range(5)] for j in range(5)] # initialize our feature relationship matrix 
    dataset_name = dataset
    dataset = Planetoid(path, name = dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    name = r'/home/jiaqing/桌面/FASG_KDD/Result/Planetoid/' + dataset_name + '_property.txt'
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
                        filepath = '/home/jiaqing/桌面/FASG_KDD/Result/Planetoid'
                        fig_name = '/' + dataset_name + '_' + embedding_method + '.txt'
                        fig_path = filepath + fig_name
                        k.to_csv(fig_path, header = None, index = None, sep = '\t')
                        #----------- save Heatmap Matrix-----------#
                        filepath = '/home/jiaqing/桌面/FASG_KDD/Result/Planetoid'
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
            # visualization
    else:

        test_case = [(1,3), (4,1)]
        embeddings = ['SAGE','GAT','GCN','GIN' ]
        #,'PubMed','Citeseer'
        for dataset, embedding_method in list(itertools.product(['Cora'],embeddings)):
            

            for (i, j) in test_case:

                ttmp = np.array([])
                best_val_acc = test_acc = 0
                t = 0

                model = Net(embedding=embedding_method)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)

                property_i = property_file.iloc[:,[i]]
                data.x = torch.tensor(np.array(property_i)).float()

                property_j = property_file.iloc[:,[j]]
                #tmp = binning(np.array(property_j), k = 6, data_len = len(data.y))
                data.y = binning(np.array(property_j), k = 6, data_len = len(data.y))

                for epoch in range(1, 3000):   
                    train()
                    train_acc, val_acc, tmp_test_acc = test()
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        test_acc = tmp_test_acc
                        embedding = model.latent
                        t = 0
                    t = t + 1
                    ttmp = np.append(ttmp, round(train_acc,3))   
                    if t > 100:
                        break
                    
                    total_epoch+=1
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    print(log.format(epoch, train_acc, best_val_acc, test_acc))
                

                # record data:
                training_error = np.append(training_error, (total_epoch, ttmp))

            #print(training_error)
        '''
        # for Cora:

        # case 1:
        seq = [4 * i for i in range(4)]
        plt.figure(figsize=(5,5))
        k = 0
        for i in seq:
            plt.plot(range(1, training_error[i]+1), training_error[i+1], label = embeddings[k])
            k+=1
        plt.show()

        # case 2:
        seq = [4 * i + 2 for i in range(4)]
        plt.figure((5,5))
        k = 0
        for i in seq:
            plt.plot(range(1, training_error[i]+1), training_error[i+1], label = embeddings[k])
            k+=1
        plt.show()

        # for Citeseer:
        # case 1:
        #plt.figure((5,5))
        # case 2:        

        '''





