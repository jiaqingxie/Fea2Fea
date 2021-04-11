import torch_geometric
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv


class augGNN(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, NTN_neurons, classes):
        super(augGNN, self).__init_()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.NTN_neurons = NTN_neurons
    
    def forward(self):
        pass

class SimpleConcat(nn.Module):
    def __init__(self, embed_1, embed_2):
        super(SimpleConcat,self).__init__()
        self.embed_1 = embed_1
        self.embed_2 = embed_2

    def forward(self):
        out = torch.cat((self.embed_1, self.embed_2), dim = 1)
        return out

class BiLinear(nn.Module):
    def __init__(self, embed_1, embed_2, embed_dim, neurons, classes):
        super(BiLinear, self).__init__()
        self.embed_1 = embed_1e
        self.embed_2 = embed_2
        self.embed_dim = embed_dim
        self.neurons = neurons
        self.classes = classes

        self.set_weights()
        self.init_param()
        
    def set_weights(self):
        self.W = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim, self.neurons))
        self.b = nn.Parameter(torch.Tensor(self.neurons,1))
    
    def init_param(self):
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.b)
    
    def forward(self):
        out = torch.mm(torch.t(embed_1), self.W.view(self.embed_dim, -1))
        out = out.view(self.dim, self.neurons)
        out = torch.mm(torch.t(out), self.embed_2)
        out = out + self.b
        return out


class NeuralTensorNetwork:
    def __init__(self, embed_1, embed_2, embed_dim, neurons, classes):
        super(BiLinear, self).__init__()
        self.embed_1 = embed_1
        self.embed_2 = embed_2
        self.embed_dim = embed_dim
        self.neurons = neurons
        self.classes = classes
        self.concat = SimpleConcat(self.embed_1, self.embed_2)

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
        out = torch.mm(torch.t(embed_1), self.W.view(self.embed_dim, -1))
        out = out.view(self.dim, self.neurons)
        out = torch.mm(torch.t(out), self.embed_2)
        comb = self.concat(embed_1, embed_2)
        out2 = torch.mm(self.V, comb)
        out = out + out2 + self.b
        return out

class GraphBlock(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, graph_conv, depth):
        super(GraphBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
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

        self.bn = nn.BatchNorm1d(embed_dim)
        if self.graph_conv == 'SAGE':
            self.conv1 = SAGEConv(1,256,normalize=True)
            self.conv2 = SAGEConv(256,64 ,normalize=True)
            self.conv3 = SAGEConv(embed_dim, embed_dim, normalize=True)
        elif self.graph_conv == 'GAT':
            self.conv1 = GATConv(1, 16,heads= 16, dropout=0.6)
            self.conv2 = GATConv(16 * 16, 64, heads=1, concat=False,
                           dropout=0.6)
            self.conv3 = GATConv(embed_dim, embed_dim, heads = 1, concat= False, dropout= 0.6)
        elif self.graph_conv == 'GCN':
            self.conv1 = GCNConv(1,256,cached=False)
            self.conv2 = GCNConv(256,64,cached=False)
            self.conv3 = GCNConv(embed_dim, embed_dim, cached=False)
        elif self.graph_conv == 'GIN':
            self.conv1 = GINConv(mlp1)
            self.conv2 = GINConv(mlp2)
            self.conv3 = GINConv(mlp3)
        else:
            pass 
        self.lin1 = nn.Linear(embed_dim,16)
        self.lin2 = nn.Linear(16,bins)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(64)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.depth):
            if i == 0:
                x = self.conv1(x, edge_index, data.edge_attr)
                if self.embedding != 'GIN':
                    x = self.batch_norm1(x)
                    x = F.relu(x)
                x = F.dropout(x, training=self.training)
            elif i == 1:
                x = self.conv2(x, edge_index, data.edge_attr)
                if self.embedding != 'GIN':
                    x = self.batch_norm2(x)
                    x = F.relu(x)
                x = F.dropout(x, training=self.training)
            else:
                if self.embedding != 'GIN':
                    x = F.relu(self.conv3(x, edge_index, data.edge_attr))
                else:
                    x = self.conv3(x, edge_index, data.edge_attr)
                x = F.dropout(x, training=self.training)
        return x
        
class SkipLastGNN:
    pass