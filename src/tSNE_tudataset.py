import numpy as np 
import pandas as pd
import os.path as osp
import statistics

import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset

from optimal_R import option, all_possible_concatenation
from graph_property import G_property, binning
from model.GNN import Net, debug_MLP
from utils import max_len_arr, tSNE_vis
from f_f_TU import train, valid, test

if __name__ == '__main__':

    paths = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
    test_case = [(1, 3)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = ['ENZYMES', 'PROTEINS', 'NCI1']
    for dataset in dataset_name:
        d_name = dataset
        data_set = TUDataset(paths + dataset, name = dataset, use_node_attr = False)
        path = r'/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/'
        train_len, valid_len= int(0.8 * len(data_set)), int(0.1 * len(data_set))  
        test_len = len(data_set) - train_len - valid_len
        batchsize = 16 if dataset != 'NCI1' else 32
        train_loader = DataLoader(data_set[0:train_len], batch_size = batchsize , shuffle=False) #### batch size 32 for NCI1
        valid_loader = DataLoader(data_set[train_len:(train_len+valid_len)], batch_size = batchsize , shuffle = False) #### batch size 32 for NCI1
        test_loader = DataLoader(data_set[(train_len+valid_len):len(data_set)], batch_size = batchsize , shuffle = False) #### batch size 32 for NCI1
        embedding = 0
        graph_embedding = 0
        for (inp, outp) in test_case:
            best_epoch = 0
            best_valid_acc = 0
            best_test_acc = 0
            op_iters = 0
            #print(tmp_txt[1][2])
            # take the optimal embedding method as graph embedding
            #print(tmp_txt[input][out])
            tmp_txt = pd.read_csv(path + d_name + '_optimal_method.txt', sep = '\t', header = None) # array
            model = Net(embedding=tmp_txt[inp][outp]).to(device) if tmp_txt[inp][outp] != 'MLP' else debug_MLP().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
            
            for epoch in range(1, 300):   
                if d_name == 'NCI1':
                    if inp == 2 or outp == 2:
                        break
                # for train
                t_loss = train(inp, outp, d_name, model, 'train', optimizer, train_loader, device)
                # for valid 
                v_acc = valid(inp, outp, d_name, model, 'valid', optimizer, valid_loader, device)
                # for test
                t_acc = test(inp, outp, d_name, model, 'test', optimizer, test_loader, device)
                print('Epoch {:03d}, Train Loss: {:.4f}, Valid acc :{:.4f}, Test acc : {:.4f}'.format(
                   epoch, t_loss, v_acc, t_acc ))

                if v_acc > best_valid_acc:
                    best_valid_acc = v_acc
                    best_test_acc = t_acc
                    best_epoch = epoch
                    # this is for loading model for predicting a batch of training set
                    model_path = '/home/jiaqing/桌面/Fea2Fea/src/model_pkl/'
                    torch.save(model, model_path + '/model_tsne_{}.pkl'.format(d_name))

                    op_iters=0

                op_iters+=1
                if op_iters > 20:
                    break
            
            for load in train_loader:
                model_path = '/home/jiaqing/桌面/Fea2Fea/src/model_pkl/'
                model = torch.load(model_path + '/model_tsne_{}.pkl'.format(d_name))
                out = model(load)
                tSNE_vis(out.linear_embed, load.y, 'mlp_embed', d_name, inp, outp, 6)
                #tSNE_vis(data.x, data.y, 'init_embed', d_name, inp, outp, 6)
                tSNE_vis(out.graph_embed, load.y, 'graph_embed', d_name, inp, outp, 6)
                break



