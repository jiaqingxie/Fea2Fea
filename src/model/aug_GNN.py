import torch_geometric
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv

class augGNN(nn.Module):
    def __init__(self, input_dim = 2, embed_dim = 64, NTN_neurons = 80, classes = 6, graph_conv = 'GIN', method = 'Bilinear'):
        super(augGNN, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.NTN_neurons = NTN_neurons
        self.method = method
        self.graph_conv = graph_conv
        self.classes = classes
        self.block = GNNBlock(1, self.embed_dim, self.graph_conv, 2)
        self.linear1 = nn.Linear(self.input_dim * self.embed_dim, self.classes)
        self.linear2 = nn.Linear(self.classes, self.classes)
        self.concat1 = SimpleConcat()
        #self.concat2 = BiLinear(64, 20)
        self.bilinear = nn.Bilinear(64, 64, 96)
        self.linear3 = nn.Linear(96,6)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr # data.x in R^N * 5 
        if self.method == 'SimpleConcat':
            tmp = x.shape[1] # less or equal than 4 if total features are 5
            x = self.block(data.x[:,[tmp-1]], edge_index, edge_attr) # 1 -> 128
            while tmp > 1:
                tmp-=1
                tt = self.block(data.x[:,[tmp-1]], edge_index, edge_attr)
                x = self.concat1(x, tt)
            #finish concatenation, go through two mlps then softmax
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return F.log_softmax(x, dim =1)
        elif self.method == 'Bilinear':
            tmp = x.shape[1] # less or equal than 4 if total features are 5
            x = self.block(data.x[:,[tmp-1]], edge_index, edge_attr) # 1 -> 64
            while tmp > 1:
                tmp-=1
                tt = self.block(data.x[:,[tmp-1]], edge_index, edge_attr)
                #self.bilinear = nn.Bilinear(x.shape[1], tt.shape[1], tt.shape[1] + 1/2 * x.shape[1])
                x = self.bilinear(x,tt)
                #x = self.concat2(x, tt)
            #finish concatenation, go through two mlps then softmax
            #self.linear1 = nn.Linear(x.shape[1], 1 /2 * x.shape[1])
            #self.linear2 = nn.Linear(1 /2 * x.shape[1], self.classes)
            x = F.relu(self.linear3(x))
            x = self.linear2(x)
            return F.log_softmax(x, dim =1)
        elif self.method == 'NTN':
            pass

class SimpleConcat(nn.Module):
    def __init__(self):
        super(SimpleConcat,self).__init__()

    def forward(self, embed_1, embed_2):
        out = torch.cat((embed_1, embed_2), dim = 1)
        return out

class BiLinear(nn.Module):
    def __init__(self, embed_dim, neurons):
        super(BiLinear, self).__init__()
        self.embed_dim = embed_dim
        self.neurons = neurons
        self.set_weights()
        self.init_param()
        
    def set_weights(self):
        self.W = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim, self.neurons))
        self.b = nn.Parameter(torch.Tensor(self.neurons,1))
    
    def init_param(self):
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.b)
    
    def forward(self, embed_1, embed_2):
        out = torch.mm(embed_1, self.W.view(self.embed_dim, -1))
        out = out.view(self.embed_dim, self.neurons)
        out = torch.mm(torch.t(out), embed_2)
        out = out + self.b
        return out

class NeuralTensorNetwork:
    def __init__(self, embed_1, embed_2, embed_dim, neurons):
        super(BiLinear, self).__init__()
        self.embed_1 = embed_1
        self.embed_2 = embed_2
        self.embed_dim = embed_dim
        self.neurons = neurons
        self.concat = SimpleConcat()

        self.set_weights()
        self.init_param()
        
    def set_weights(self):
        self.W = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim, self.neurons))
        self.V = nn.Parameter(torch.Tensor(self.neurons, 2 * self.embed_dim))
        self.b = nn.Parameter(torch.Tensor(self.neurons,1))
    
    def init_param(self):
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.b)
    
    def forward(self):
        print(torch.t(embed_1).shape)
        print( self.W.view(self.embed_dim, -1).shape)
        out = torch.mm(torch.t(embed_1), self.W.view(self.embed_dim, -1))
        out = out.view(self.dim, self.neurons)
        out = torch.mm(torch.t(out), self.embed_2)
        comb = self.concat(embed_1, embed_2)
        out2 = torch.mm(self.V, comb)
        out = out + out2 + self.b
        return out

class GNNBlock(nn.Module):
    def __init__(self, input_dim,  embed_dim, graph_conv, depth):
        super(GNNBlock, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.graph_conv = graph_conv
        self.depth = depth

        mlp1 = nn.Sequential(
                nn.Linear(1, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,128),
            )
        mlp2 = nn.Sequential(
                nn.Linear(128,64 ),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64,64),
            )
        mlp3 = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(self.embed_dim)
        if self.graph_conv == 'SAGE':
            self.conv1 = SAGEConv(1,256,normalize=True)
            self.conv2 = SAGEConv(256,64 ,normalize=True)
            self.conv3 = SAGEConv(self.embed_dim, self.embed_dim, normalize=True)
        elif self.graph_conv == 'GAT':
            self.conv1 = GATConv(1, 16,heads= 16, dropout=0.6)
            self.conv2 = GATConv(16 * 16, 64, heads=1, concat=False,
                           dropout=0.6)
            self.conv3 = GATConv(self.embed_dim, self.embed_dim, heads = 1, concat= False, dropout= 0.6)
        elif self.graph_conv == 'GCN':
            self.conv1 = GCNConv(1,256,cached=False)
            self.conv2 = GCNConv(256,64,cached=False)
            self.conv3 = GCNConv(self.embed_dim, self.embed_dim, cached=False)
        elif self.graph_conv == 'GIN':
            self.conv1 = GINConv(mlp1)
            self.conv2 = GINConv(mlp2)
            self.conv3 = GINConv(mlp3)
        else:
            pass 
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(64)

    def forward(self, data, edge_index, edge_attr):
        x = data
        for i in range(self.depth):
            if i == 0:
                x = self.conv1(x, edge_index, edge_attr)
                if self.graph_conv != 'GIN':
                    x = self.batch_norm1(x)
                    x = F.relu(x)
                x = F.dropout(x, training=self.training)
            elif i == 1:
                x = self.conv2(x, edge_index, edge_attr)
                if self.graph_conv != 'GIN':
                    x = self.batch_norm2(x)
                    x = F.relu(x)
                x = F.dropout(x, training=self.training)
            else:
                if self.graph_conv != 'GIN':
                    x = F.relu(self.conv3(x, edge_index))
                else:
                    x = self.conv3(x, edge_index, edge_attr)
                x = F.dropout(x, training=self.training)
        return x
        
class SkipLastGNN:
    pass


if __name__ == '__main__':
    model = augGNN()