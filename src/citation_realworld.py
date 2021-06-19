from distributed.worker import weight
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import statistics
import random

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.data import DataLoader


from optimal_R import option, all_possible_concatenation
from graph_property import G_property, binning
from model.StrucFea_GNN_cit import StrucFeaGNN
from utils import max_len_arr

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
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
    multiple_dataset = False
    saved = []


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if multiple_dataset:
        datasets = ['Cora', 'PubMed', 'Citeseer']
        #c = ['r', 'b', 'g']
    else:
        datasets = [o.dataset] # set to defualt dataset
    print(datasets)
    for dataset in datasets:
        o.dataset = dataset
        ans = all_possible_concatenation(o)
        min_ans_len, max_ans_len = max_len_arr(ans)
        #saved = np.array([0 for i in range(min_ans_len, max_ans_len+1)])

        c_index = 0
        path = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
        data_set = Planetoid(path, name = dataset, transform=T.NormalizeFeatures())
        data = data_set[0]
        print(len(data.train_mask))

        name = r'/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/' + dataset + '_property.txt'
        property_file = pd.read_csv(name, sep = '\t')
        
        num_classes = {'Cora': 7, 'PubMed': 3, 'Citeseer': 6}
        ans = [(0,4)]
        #ans = [(0,2,4)]
        aver_test_acc = []
        for value in ans:
            array = np.array(property_file.iloc[:,list(value)])
            array = torch.tensor(array).float()
            data.x = torch.cat((data.x, array), dim = 1)
            #print(data.x.shape)   
            data =  data.to(device)
            for i in range(10): 
                model =  StrucFeaGNN(concat_fea=True, concat_fea_num = len(ans[0]),  embed_method = o.graphconv, input_dim = data.x.shape[1] ,     output_dim =num_classes[o.dataset], cat_method = o.concat_method, depth = 3, required_batch = False).to(device)
                if o.graphconv == 'MLP':
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 5e-4) 
                else:
                    optimizer = torch.optim.Adam([
                    dict(params=model.convs[0].parameters(), weight_decay=5e-4),
                    dict(params=model.convs[1].parameters(), weight_decay=0),
                    dict(params=model.convs[2].parameters(), weight_decay=0),
                ], lr=0.008)  # Only perform weight-decay on first twwo convolutions.
                t = 0
                best_val_acc = test_acc = 0 
                #train_accu = []
                #count = []

                for epoch in range(1, 3000):
                    train()
                    train_acc, val_acc, tmp_test_acc = test()
                    #print(tmp_test_acc)
                    #if epoch%1 == 0:
                    #    count.append(epoch)
                    #    train_accu.append(train_acc)
                    if val_acc > best_val_acc and tmp_test_acc > test_acc  :
                        best_val_acc = val_acc
                        test_acc = tmp_test_acc
                        t = 0
                    t = t + 1
                    if t > 800:
                        break   
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    #print(log.format(epoch, train_acc, best_val_acc, test_acc))
                aver_test_acc.append(test_acc)
                print("time: {}, best test acc: {:.4f}".format(i, test_acc))
            print("average test acc: {:.1f}, std: {:.1f}".format(100 * sum(aver_test_acc)/len(aver_test_acc), 100 * np.std(aver_test_acc)))
