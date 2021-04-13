import torch
import numpy as np 
import pandas as pd
from optimal_R import option, all_possible_concatenation
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from model.aug_GNN import augGNN
import torch.nn.functional as F
from graph_property import G_property, binning

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


if __name__ == '__main__':
    o = option()
    ans = all_possible_concatenation(o)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # loading property matrix
    print(o.dataset)
    path = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
    dataset = Planetoid(path, name = o.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    # read property file as the input of graph data
    name = r'/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/' + o.dataset + '_property.txt'
    property_file = pd.read_csv(name, sep = '\t')
    data.x = np.array(property_file.iloc[:,1:3])
    data.x = torch.tensor(data.x).float()
    data.y = np.array(property_file.iloc[:,[4]])
    print(property_file.iloc[:,[1]])
    number = len(data.y)
    data.y = binning(data.y, k = 6,data_len =  number)
    model =  augGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)

    data =  data.to(device)

    t = 0
    best_val_acc = test_acc = 0 
    for epoch in range(1, 3000):
        
        train()
        train_acc, val_acc, tmp_test_acc = test()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

            t = 0
        t = t + 1
        if t > 1200:
            break   
        
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))