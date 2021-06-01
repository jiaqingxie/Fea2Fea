import torch_geometric
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv

class augGNN(nn.Module):
    def __init__(self, input_dim = 2, embed_dim = 64, NTN_neurons = 64, classes = 6, graph_conv = 'GIN', method = 'SimpleConcat'):
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
        self.concat1 = nn.ModuleList()
        self.concat2 = nn.ModuleList()
        self.concat3 = nn.ModuleList()
        self.in1_features = self.embed_dim
        for i in range(input_dim-1):
            self.concat1.append(SimpleConcat())
            self.concat2.append(nn.Bilinear(int(self.in1_features), int(self.embed_dim), int(1/2 * self.in1_features + self.embed_dim)))
            self.concat3.append(NeuralTensorNetwork(self.embed_dim, self.NTN_neurons))
            self.in1_features = int(1/2 * self.in1_features + self.embed_dim)
        self.linear3 = nn.Linear(self.in1_features,self.classes)
        self.linear4 = nn.Linear(embed_dim, 6)

        self.graph_embed = 0 
        self.linear_embed = 0

    def forward(self, data):
        x, edge_index  = data.x, data.edge_index# data.x in R^N * 5 
        if self.method == 'SimpleConcat':
            tmp = x.shape[1] # less or equal than 4 if total features are 5
            i = 0
            x = self.block(data.x[:,[tmp-1]], edge_index) # 1 -> 128
            while tmp > 1:
                tmp-=1
                tt = self.block(data.x[:,[tmp-1]], edge_index)
                x = self.concat1[i](x, tt)
                i+=1
            self.graph_embed = x
            #finish concatenation, go through two mlps then softmax
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            self.linear_embed = x
            return F.log_softmax(x, dim =1)
        elif self.method == 'Bilinear':
            tmp = x.shape[1] # less or equal than 4 if total features are 5
            i = 0
            x = self.block(data.x[:,[tmp-1]], edge_index) # 1 -> 64
            while tmp > 1:
                tmp-=1
                tt = self.block(data.x[:,[tmp-1]], edge_index)
                x = self.concat2[i](x,tt)
                i+=1
            #finish concatenation, go through two mlps then softmax 
            self.graph_embed = x
            x = F.relu(self.linear3(x))
            x = self.linear2(x)
            self.linear_embed = x
            return F.log_softmax(x, dim =1)
        elif self.method == 'NTN':
            tmp = x.shape[1] # less or equal than 4 if total features are 5
            i = 0
            x = self.block(data.x[:,[tmp-1]], edge_index) # 1 -> 64
            while tmp > 1:
                tmp-=1
                tt = self.block(data.x[:,[tmp-1]], edge_index)
                x = self.concat3[i](x,tt)
                i+=1
            self.graph_embed = x
            #finish concatenation, go through two mlps then softmax 
            x = F.relu(self.linear4(x))
            x = self.linear2(x)
            self.linear_embed = x
            return F.log_softmax(x, dim =1)

class SimpleConcat(nn.Module):
    def __init__(self):
        super(SimpleConcat,self).__init__()

    def forward(self, embed_1, embed_2):
        out = torch.cat((embed_1, embed_2), dim = 1)
        return out

class NeuralTensorNetwork(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(NeuralTensorNetwork, self).__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.concat = SimpleConcat()

        self.setup_weights()
        self.init_parameters()
        
    def setup_weights(self):
        self.W = nn.Parameter(torch.Tensor(self.output_dim, self.embed_dim, self.embed_dim))
        self.V = nn.Parameter(torch.Tensor(2 * self.embed_dim, self.output_dim))
        self.b = nn.Parameter(torch.Tensor(1, self.embed_dim))
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.b)
    
    def forward(self, embed_1, embed_2):
        e1 = embed_1
        e2 = embed_2
        #print(e1.shape)
        batch_size = e1.shape[0]
        k = self.output_dim
        feed_forward_product = torch.mm(self.concat(e1,e2), self.V) # V*[e1,e2]
        

        bilinear_tensor_product = []

        for i in range(k):
            btp = torch.sum((e2 * torch.mm(e1, self.W[i]))+ self.b, dim=1)
            bilinear_tensor_product.append(btp)


       # print(torch.cat(bilinear_tensor_product).shape)
        #print(batch_size)
        out = F.tanh(torch.reshape(torch.cat(bilinear_tensor_product, dim=0), (batch_size,k)) + feed_forward_product)
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

    def forward(self, data, edge_index):
        x = data
        for i in range(self.depth):
            if i == 0:
                x = self.conv1(x, edge_index)
                if self.graph_conv != 'GIN':
                    x = self.batch_norm1(x)
                    x = F.relu(x)
                x = F.dropout(x, training=self.training)
            elif i == 1:
                x = self.conv2(x, edge_index)
                if self.graph_conv != 'GIN':
                    x = self.batch_norm2(x)
                    x = F.relu(x)
                x = F.dropout(x, training=self.training)
            else:
                if self.graph_conv != 'GIN':
                    x = F.relu(self.conv3(x, edge_index))
                else:
                    x = self.conv3(x, edge_index)
                x = F.dropout(x, training=self.training)
        return x
        
class SkipLastGNN:
    pass


if __name__ == '__main__':
    model = augGNN()