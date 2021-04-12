import torch
import numpy as np 
import pandas as pd
from optimal_R import option, all_possible_concatenation
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from model.aug_GNN import augGNN

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
    # loading property matrix
    print(o.dataset)
    path = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
    dataset = Planetoid(path, name = o.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    # read property file as the input of graph data
    name = r'/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/' + o.dataset + '_property.txt'
    property_file = pd.read_csv(name, sep = '\t')
    model =  Net(embedding=embedding_method).to(device)


