import numpy as np 
import pandas as pd
import os.path as osp
import statistics

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F

from optimal_R import option, all_possible_concatenation
from graph_property import G_property, binning
from model.aug_GNN import augGNN
from utils import max_len_arr, tSNE_vis

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
    paths = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
    

    dataset_name = ['Cora', 'PubMed', 'Citeseer']
    for dataset in dataset_name:
        o.dataset = dataset
        ans = all_possible_concatenation(o)
        d_name = dataset
        dataset = Planetoid(paths, name = dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        path = r'/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/'
        name = path + d_name + '_property.txt'
        property_file = pd.read_csv(name, sep = '\t')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for case in ans:
    
            property_i = np.array(property_file.iloc[:,list(case)])
            data.x = torch.tensor(property_i).float()
            #print(data.x.shape)
            property_j = np.array(property_file.iloc[:,[o.aim_feature]])
            data.y = binning(property_j, k = 6, data_len = len(data.y))
            
            embedding = 0
            best_val_acc = test_acc = 0
            t = 0
            train_accu_plot = []
            epoch_plot = []
            model = augGNN(input_dim = len(case), method = 'NTN').to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.015, weight_decay=1e-4)
            data =  data.to(device)
            
            for epoch in range(1, 3000):   
                train()
                train_acc, val_acc, tmp_test_acc = test()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                    embedding = model.linear_embed # best validation
                    graph_embedding = model.graph_embed 
                    t = 0
                t = t + 1
                if t > 400:
                    break   
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                # debug:
                # print(log.format(epoch, train_acc, best_val_acc, test_acc))

            nb_classes = 6
            confusion_matrix = torch.zeros(nb_classes,nb_classes)
            
            tSNE_vis(embedding, data.y, 'mlp_embed_aug', d_name, case, o.aim_feature, 6)
            tSNE_vis(data.x, data.y, 'init_embed', d_name, case, o.aim_feature, 6)
            tSNE_vis(graph_embedding, data.y, 'graph_embed_aug', d_name, case, o.aim_feature, 6)
            break # test on first element of all possible combination results
           


