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
from model.aug_GNN import augGNN
from utils import max_len_arr, tSNE_vis
from f_f_TU import valid, test

def train(i, j, dn, model, task, optimizer, train_loader, device, k = 6):
    total_loss = 0
    model.train()
    total_num_nodes = 0
    t= 0
    graph_embed = 0
    linear_embed = 0

    for load in train_loader:
        name = r'/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/' + dn + '/' + dn + '_property' + str(t) + task +'.txt'
        property_file = pd.read_csv(name, sep = '\t')
        propert_i = property_file.iloc[:,list(i)] if isinstance(i,tuple) else property_file.iloc[:,[i]]
        array = np.array(propert_i)
        load.x = torch.tensor(array).float()

        propert_j = property_file.iloc[:,[j]]
        array_2 = np.array(propert_j)
        number = len(array_2)
        load.y = binning(array_2, k = k, data_len =  number)
        # --------- training loop ---------- #
        
        load = load.to(device)
        optimizer.zero_grad()
        out = model(load)
        loss = F.nll_loss(out,load.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(load.y)
        total_num_nodes+=len(load.y)
        if t == 0:
            graph_embed = model.graph_embed
            linear_embed = model.linear_embed
        t+=1

        #print(loss)
    train_loss = total_loss / total_num_nodes
    return train_loss, graph_embed, linear_embed




def train_tsne(i, j, dn, l_m, g_m, task, train_loader, device, k = 6):
    t = 0
    for load in train_loader:
        name = r'/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/' + dn + '/' + dn + '_property' + str(t) + task +'.txt'
        property_file = pd.read_csv(name, sep = '\t')
        propert_i = property_file.iloc[:,list(i)] if isinstance(i,tuple) else property_file.iloc[:,[i]]
        array = np.array(propert_i)
        load.x = torch.tensor(array).float()

        propert_j = property_file.iloc[:,[j]]
        array_2 = np.array(propert_j)
        number = len(array_2)
        load.y = binning(array_2, k = k, data_len =  number)

        load = load.to(device)
        #out = model(load)
        tSNE_vis(load.x, load.y, 'init_embed_aug', d_name, value, o.aim_feature, 6)
        tSNE_vis(l_m, load.y, 'mlp_embed_aug', d_name, value, o.aim_feature, 6)
                #tSNE_vis(data.x, data.y, 'init_embed', d_name, inp, outp, 6)
        tSNE_vis(g_m, load.y, 'graph_embed_aug', d_name, value, o.aim_feature, 6)
        break


if __name__ == '__main__':
    o = option()
    paths = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
    test_case = [(1, 3)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = ['ENZYMES', 'PROTEINS', 'NCI1']
    for dataset in dataset_name:
        o.dataset = dataset
        ans = all_possible_concatenation(o)
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
        for value in ans:
            best_epoch = 0
            best_valid_acc = 0
            best_test_acc = 0
            op_iters = 0

            model = augGNN(input_dim = len(value), method = 'NTN').to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
            
            best_linear_embed = 0
            best_graph_embed = 0

            for epoch in range(1, 300):   
                if d_name == 'NCI1':
                    if o.aim_feature == 2:
                        break
                # for train
                t_loss, graph_embed, linear_embed = train(value, o.aim_feature, d_name, model, 'train', optimizer, train_loader, device)
                # for valid 
                v_acc = valid(value, o.aim_feature, d_name, model, 'valid', optimizer, valid_loader, device)
                # for test
                t_acc = test(value, o.aim_feature, d_name, model, 'test', optimizer, test_loader, device)
                #print('Epoch {:03d}, Train Loss: {:.4f}, Valid acc :{:.4f}, Test acc : {:.4f}'.format(
                 #  epoch, t_loss, v_acc, t_acc ))

                if v_acc > best_valid_acc:
                    best_valid_acc = v_acc
                    best_test_acc = t_acc
                    best_epoch = epoch
                    best_linear_embed = linear_embed
                    best_graph_embed = graph_embed
                    # this is for loading model for predicting a batch of training set
                    #model_path = '/home/jiaqing/桌面/Fea2Fea/src/model_pkl/'
                    #torch.save(model, model_path + '/model_tsne_{}.pkl'.format(d_name))

                    op_iters=0

                op_iters+=1
                if op_iters > 20:
                    break



            #model_path = '/home/jiaqing/桌面/Fea2Fea/src/model_pkl/'
            #model = torch.load(model_path + '/model_tsne_{}.pkl'.format(d_name))
            #model.to(device)
            print("visualizing embeddings...")
            train_tsne(value, o.aim_feature, d_name, best_linear_embed, best_graph_embed, 'train', train_loader, device, k = 6)
            break


