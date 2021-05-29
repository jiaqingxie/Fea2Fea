import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import statistics
import matplotlib.colors as colors
from sklearn.manifold import TSNE

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F

from optimal_R import option, all_possible_concatenation
from graph_property import G_property, binning
from model.GNN import Net, debug_MLP
from utils import max_len_arr, tSNE_vis

def train():
    model.train()
    optimizer.zero_grad     ()
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
    paths = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
    test_case = [(1, 3)]

    dataset_name = ['Cora', 'PubMed', 'Citeseer']
    for dataset in dataset_name:
        d_name = dataset
        dataset = Planetoid(paths, name = dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        path = r'/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/'
        name = path + d_name + '_property.txt'
        property_file = pd.read_csv(name, sep = '\t')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for (inp, outp) in test_case:
    
            property_i = np.array(property_file.iloc[:,[inp]])
            data.x = torch.tensor(property_i).float()

            property_j = np.array(property_file.iloc[:,[outp]])
            tmp = binning(property_j, k = 6, data_len = len(data.y))
            data.y = binning(property_j, k = 6, data_len = len(data.y))
            
            # find optimal graph embedding method according to each
            # input graph feature and output graph feature
            tmp_txt = pd.read_csv(path + d_name + '_optimal_method.txt', sep = '\t', header = None) # array
        
            print(tmp_txt.iloc[inp,outp])
            embedding = 0
            best_val_acc = test_acc = 0
            t = 0
            train_accu_plot = []
            epoch_plot = []
            #print(tmp_txt[1][2])
            # take the optimal embedding method as graph embedding
            #print(tmp_txt[input][out])
            model = Net(embedding='GIN').to(device) if tmp_txt[inp][outp] != 'MLP' else debug_MLP().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
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
                #print(log.format(epoch, train_acc, best_val_acc, test_acc))

            nb_classes = 6
            confusion_matrix = torch.zeros(nb_classes,nb_classes)
            
            tSNE_vis(embedding, data.y, 'mlp_embed', d_name, inp, outp, 6)
            #tSNE_vis(data.x, data.y, 'init_embed', d_name, inp, outp, 6)
            tSNE_vis(graph_embedding, data.y, 'graph_embed', d_name, inp, outp, 6)

            '''
            if you want to print f1 score, then uncomment this part
            pre_comb = torch.tensor([])
            real_comb = torch.tensor([])
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

           


