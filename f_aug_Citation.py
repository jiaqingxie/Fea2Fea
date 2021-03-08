import itertools
import os.path as osp
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch_geometric.datasets import Planetoid,TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn import ARMAConv,AGNNConv
from torch_geometric.nn import GINConv,GATConv,GCNConv
from torch_geometric.nn import SAGEConv,SplineConv
import math
import matplotlib.pyplot as plt
from graph_property import G_property, binning
import seaborn as sns
from sklearn.metrics import f1_score


from sklearn.preprocessing import StandardScaler     ### Three data preprocessing methods
from sklearn.preprocessing import Normalizer         
from sklearn.preprocessing import MinMaxScaler


path = osp.join( 'C:\\Users\\11415\\Desktop\\gnn\\data\\')
dataset = 'PubMed'
dataset = Planetoid(path, name = dataset, transform=T.NormalizeFeatures())
#dataset = Planetoid(path, name = dataset, transform=None)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.x.shape[1], 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    '''
    def __init__(self):
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
        #self.conv1 = SAGEConv(data.x.shape[1],64,normalize=True)
        #self.conv1 = SAGEConv(data.x.shape[1], 64, normalize=True)
        #self.conv2 = SAGEConv(64,32,normalize=True)
        #self.conv1 = GATConv(data.x.shape[1], 16,heads= 16, dropout=0.6)
        #self.conv2 = GATConv(16 * 16, 32, heads=1, concat=False,
                           #dropout=0.6)

        #self.conv3 = SAGEConv(1,64,normalize=True)
        #self.conv4 = SAGEConv(64,3,normalize=True) 
        self.conv1 = GCNConv(data.x.shape[1],64,cached=True)
        self.conv2 = GCNConv(64,32,cached=True)
        self.lin1 = nn.Linear(32,18)  
        self.lin2 = nn.Linear(18,7)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    def forward(self):
        #x2, x1, edge_index = data.x[:,:data.x.shape[1]], data.x[:,data.x.shape[1]], data.edge_index
        #x = F.relu(self.conv1(x2, edge_index))
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        #self.latent = graph_embedding
        
        x_ = F.relu(self.conv3(x1, edge_index))
        x_ = F.dropout(x_, training=self.training)
        x_ = F.relu(self.conv4(x_, edge_index))
        x_ = F.dropout(x_, training=self.training)
        
        #x = torch.cat((x, x_), dim = 1)
        
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        
        return F.log_softmax(x, dim =1)
        '''
nor_x = Normalizer() 
std_x = StandardScaler()

kk1 = pd.read_csv("C:\\Users\\11415\\Desktop\\gnn\\result\\Planetoid_property\\PubMed_property.txt",sep = '\t')
print(kk1)
kk = np.array([kk1['Aver_path_len']])


bb = np.array([kk1['Clustering_coefficient']])
kk = nor_x.fit_transform(kk).T
bb = nor_x.fit_transform(bb).T
#kk = np.array([kk['Aver_path_len']]).T
#kk = std_x.fit_transform(kk)


for i in range(1,50):
    data.x = torch.cat((data.x, torch.tensor(kk).float()),dim = 1)
    data.x = torch.cat((data.x, torch.tensor(bb).float()),dim = 1)


#data.x = nor_x.fit_transform(data.x)
#data.x = torch.tensor(data.x).float()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =  Net().to(device)
data =  data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)



print(data.x)

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

best_val_acc = test_acc = 0
t = 0

train_accu_plot = []
epoch_plot = []

for epoch in range(1, 10000):
                   
    train()
    train_acc, val_acc, tmp_test_acc = test()
    train_accu_plot.append(train_acc)
    epoch_plot.append(epoch)
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        t = 0
    t = t + 1
    if t > 2000:
        break   
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

plt.plot(epoch_plot,train_accu_plot)
plt.show()

nb_classes = 7
confusion_matrix = torch.zeros(nb_classes,nb_classes)

logits, accs = model(), []
for _, mask in data('test_mask'):
    pred = logits[mask].max(1)[1]
    print(len(pred))
    print(len(data.y[mask]))
    for i in range(len(pred)):
        confusion_matrix[pred[i]][data.y[mask][i]] = confusion_matrix[pred[i]][data.y[mask][i]]+1
    print(f1_score(data.y[mask].cpu(), pred.cpu(), average='macro'))
print(confusion_matrix)#
import itertools
import os.path as osp
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch_geometric.datasets import Planetoid,TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn import ARMAConv,AGNNConv
from torch_geometric.nn import GINConv,GATConv,GCNConv
from torch_geometric.nn import SAGEConv,SplineConv
import math
import matplotlib.pyplot as plt
from graph_property import G_property, binning
import seaborn as sns
from sklearn.metrics import f1_score


from sklearn.preprocessing import StandardScaler     ### Three data preprocessing methods
from sklearn.preprocessing import Normalizer         
from sklearn.preprocessing import MinMaxScaler


path = osp.join( 'C:\\Users\\11415\\Desktop\\gnn\\data\\')
dataset = 'PubMed'
dataset = Planetoid(path, name = dataset, transform=T.NormalizeFeatures())
#dataset = Planetoid(path, name = dataset, transform=None)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.x.shape[1], 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    '''
    def __init__(self):
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
        #self.conv1 = SAGEConv(data.x.shape[1],64,normalize=True)
        #self.conv1 = SAGEConv(data.x.shape[1], 64, normalize=True)
        #self.conv2 = SAGEConv(64,32,normalize=True)
        #self.conv1 = GATConv(data.x.shape[1], 16,heads= 16, dropout=0.6)
        #self.conv2 = GATConv(16 * 16, 32, heads=1, concat=False,
                           #dropout=0.6)

        #self.conv3 = SAGEConv(1,64,normalize=True)
        #self.conv4 = SAGEConv(64,3,normalize=True) 
        self.conv1 = GCNConv(data.x.shape[1],64,cached=True)
        self.conv2 = GCNConv(64,32,cached=True)
        self.lin1 = nn.Linear(32,18)  
        self.lin2 = nn.Linear(18,7)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    def forward(self):
        #x2, x1, edge_index = data.x[:,:data.x.shape[1]], data.x[:,data.x.shape[1]], data.edge_index
        #x = F.relu(self.conv1(x2, edge_index))
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        #self.latent = graph_embedding
        
        x_ = F.relu(self.conv3(x1, edge_index))
        x_ = F.dropout(x_, training=self.training)
        x_ = F.relu(self.conv4(x_, edge_index))
        x_ = F.dropout(x_, training=self.training)
        
        #x = torch.cat((x, x_), dim = 1)
        
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        
        return F.log_softmax(x, dim =1)
        '''
nor_x = Normalizer() 
std_x = StandardScaler()

kk1 = pd.read_csv("C:\\Users\\11415\\Desktop\\gnn\\result\\Planetoid_property\\PubMed_property.txt",sep = '\t')
print(kk1)
kk = np.array([kk1['Aver_path_len']])


bb = np.array([kk1['Clustering_coefficient']])
kk = nor_x.fit_transform(kk).T
bb = nor_x.fit_transform(bb).T
#kk = np.array([kk['Aver_path_len']]).T
#kk = std_x.fit_transform(kk)


for i in range(1,50):
    data.x = torch.cat((data.x, torch.tensor(kk).float()),dim = 1)
    data.x = torch.cat((data.x, torch.tensor(bb).float()),dim = 1)


#data.x = nor_x.fit_transform(data.x)
#data.x = torch.tensor(data.x).float()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =  Net().to(device)
data =  data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)



print(data.x)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

def test():
    model.eval()
    with torch.no_grad():
        logits, accs = model(), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs

best_val_acc = test_acc = 0
t = 0

train_accu_plot = []
epoch_plot = []

for epoch in range(1, 10000):
                   
    train()
    train_acc, val_acc, tmp_test_acc = test()
    train_accu_plot.append(train_acc)
    epoch_plot.append(epoch)
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        t = 0
    t = t + 1
    if t > 2000:
        break   
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

plt.plot(epoch_plot,train_accu_plot)
plt.show()

nb_classes = 7
confusion_matrix = torch.zeros(nb_classes,nb_classes)

logits, accs = model(), []
for _, mask in data('test_mask'):
    pred = logits[mask].max(1)[1]
    print(len(pred))
    print(len(data.y[mask]))
    for i in range(len(pred)):
        confusion_matrix[pred[i]][data.y[mask][i]] = confusion_matrix[pred[i]][data.y[mask][i]]+1
    print(f1_score(data.y[mask].cpu(), pred.cpu(), average='macro'))
print(confusion_matrix)#
