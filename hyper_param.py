from argparse import ArgumentParser
from model.var_GNN import Net
import numpy as np 
import pandas as pd
import os.path as osp
import torch 
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from graph_property import binning, G_property

def train(task):
    if task == 'node':
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()
    elif task == 'graph':
        pass


def test():
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def option():
    parser = ArgumentParser()
    parser.add_argument('--input_feature', default = 1, type = int, help = 'input feature')
    parser.add_argument('--output_feature', default = 2, type = int, help = 'output feature')
    parser.add_argument('--task', default = 'node', type = str, help = 'node / graph dataset')
    parser.add_argument('--dataset', default = 'Cora', type = str, help = 'dataset name')
    parser.add_argument('--hyperparameter', default = 'binning', type = str, help = 'hyper-para task')
    parser.add_argument('--min_bins', default = 2, type = int, help = 'minimum number of bins')
    parser.add_argument('--max_bins', default = 6, type = int, help = 'maximum number of bins')
    parser.add_argument('--min_depth', default = 2, type = int, help = 'minimum depth of GNN architecture')
    parser.add_argument('--max_depth', default = 6, type = int, help = 'maximum depth of GNN architecture')
    parser.add_argument('--hidden_dim', default = 2, type = int, help = 'hidden dimension')
    parser.add_argument('--batchnorm', default = 0, type = bool, help = 'if BatchNorm')
    parser.add_argument('--embedding', default = 'GIN', type = str, help = 'graph embedding method')
    return parser.parse_args()

if __name__ == "__main__":
    a = option() # add parser
    path = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
    device = 
    # if node dataset
    if a.task == 'node':
        dataset = Planetoid(path, name = a.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        # we just consider planetoid dataset
        # if you want to try on other small node datasets, be sure that they are downloaded to /../data folder
        name = r'/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/' + a.dataset + '_property.txt'
        property_file = pd.read_csv(name, sep = '\t')
        #print(property_file.shape)
        # input property
        propert_i = property_file.iloc[:,[a.input_feature]]
        array = np.array(propert_i)
        data.x = torch.tensor(array).float()
        #print(data.x.shape)
        propert_j = property_file.iloc[:,[a.output_feature]]
        array_2 = np.array(propert_j)
        if a.hyperparameter == 'binning':
            for bins in range(a.min_bins, a.max_bins +1):
                data.y = binning(array_2, k = bins, data_len =  len(data.y))
                # to GPU
                model = Net(bins = bins).to(device)
                data =  data.to(device)
                # optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=0.04, weight_decay=5e-4)    
                # training epoch
                for epoch in range(1, 3000):


            

        elif a.hyperparameter == 'depth':
            pass
    
    # else if graph dataset
    elif a.task == 'graph':
        pass
    