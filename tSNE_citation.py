from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os.path as osp
import torch_geometric.transforms as T
import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from graph_property import G_property, binning
from torch_geometric.nn import GINConv,GCNConv
from torch_geometric.datasets import Planetoid
import matplotlib.colors as colors

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
            self.conv1 = SAGEConv(1,256,normalize=True)
            self.conv2 = SAGEConv(256,64 ,normalize=True)
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
        x = F.relu(self.conv1(x, edge_index, data.edge_attr))
        x = F.dropout(x, training=self.training)
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

if __name__ == "__main__":
    
    path = osp.join('/home/jiaqing/桌面/FASG_KDD/data/')
    test_case = [(2, 1),(1, 3)]


    dataset_name = ['Cora', 'PubMed', 'Citeseer']
    for dataset in dataset_name:
        d_name = dataset
        dataset = Planetoid(path, name = dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        name = r'/home/jiaqing/桌面/FASG_KDD/Result/Planetoid/' + d_name + '_property.txt'
        property_file = pd.read_csv(name, sep = '\t')
        for (i, j) in test_case:
            embedding = 0
            best_val_acc = test_acc = 0
            t = 0
            train_accu_plot = []
            epoch_plot = []

            model = Net(embedding="GIN")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)

            property_i = property_file.iloc[:,[i]]
            data.x = torch.tensor(np.array(property_i)).float()

            property_j = property_file.iloc[:,[j]]
            tmp = binning(np.array(property_j), k = 6, data_len = len(data.y))
            data.y = binning(np.array(property_j), k = 6, data_len = len(data.y))

            for epoch in range(1, 3000):   
                train()
                train_acc, val_acc, tmp_test_acc = test()
                #train_accu_plot.append(train_acc)
                #epoch_plot.append(epoch)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                    embedding = model.latent
                    t = 0
                t = t + 1
                if t > 100:
                    break   
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, train_acc, best_val_acc, test_acc))

            nb_classes = 6
            confusion_matrix = torch.zeros(nb_classes,nb_classes)
            pre_comb = torch.tensor([])
            real_comb = torch.tensor([])

            '''
            #----- print macro-f1 score
            with torch.no_grad():
                logits, accs = model(), []
                for _, mask in data('test_mask'):
                    pred = logits[mask].max(1)[1]
                    pre_comb = torch.cat((pre_comb, pred), 0)
                    real_comb = torch.cat((real_comb, data.y[mask]), 0)

                    #print(pred)
                    #print(data.y[mask])
                    for i in range(len(pred)):
                        confusion_matrix[pred[i]][data.y[mask][i]] = confusion_matrix[pred[i]][data.y[mask][i]]+1
                print(confusion_matrix)#
                print(f1_score(pre_comb.numpy(), real_comb.numpy(), average='macro'))
            '''

            # draw tSNE pictures here:
            x = embedding.detach().numpy()
            #y = np.array(property_j)
            X_tsne = TSNE(n_components=2,random_state=33).fit_transform(x)
            plt.figure(figsize=(6, 6))
            ax = plt.subplot(1,1,1,)

            values = range(6)
            cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
            scaMap = plt.cm.ScalarMappable(norm = cNorm  ,cmap = "coolwarm")

            for k in range(6):  
                colorval = scaMap.to_rgba(values[k])
                ax.scatter(X_tsne[np.where(tmp.numpy() == k), 0], X_tsne[np.where(tmp.numpy() == k), 1] ,label = k, s =3, color = colorval)


            handles,labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper right',fontsize = 7)
            plt.xlabel("tSNE 1",fontsize = 12)
            plt.ylabel("tSNE 2", fontsize = 12)
            plt.tick_params(labelsize=12)
            name2 = r'/home/jiaqing/桌面/FASG_KDD/Result/tSNE/'
            plt.savefig(name2 + str(d_name)+"_"+ str(i)+ "to" + str(j) +"_tSNE.eps", dpi = 800, format = 'eps')
            #plt.show()
            #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=tmp.numpy(), cmap = "rainbow")
            #plt.legend()

            
