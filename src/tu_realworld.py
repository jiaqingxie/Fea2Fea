import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import statistics
import random

import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset

from optimal_R import option, all_possible_concatenation
from graph_property import G_property, binning
from model.aug_GNN import augGNN
from model.StrucFea_GNN import StrucFeaGNN
from utils import max_len_arr

def print_conf_mtx(dn, task, model, test_loader, device, num_class):
    model.eval()
    total_num_nodes = 0
    t = 0
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    conf_matrix = torch.zeros(num_class, num_class)
    correct = 0
    for load in test_loader:
        name = r'/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/' + dn + '/' + dn + '_property' + str(t) + task +'.txt'
        property_file = pd.read_csv(name, sep = '\t')
        propert_i = property_file.iloc[:,list(i)] if isinstance(i,tuple) else property_file.iloc[:,[i]]
        array = np.array(propert_i)
        load.x = torch.cat((load.x, torch.tensor(array).float()), dim = 1)

        with torch.no_grad():
            load = load.to(device)
            pred = model(load, load.batch).max(dim=1)[1]
            print(pred)
            print(load.y)
            for p, l in zip(pred, load.y):
                p = p.long()
                l = l.long()
                conf_matrix[p, l] += 1
        correct += pred.eq(load.y).sum().item()
        t+=1
    
    print(correct / len(test_loader.dataset))
    
    print(conf_matrix)

def reserve(task, dn, loader, folds):
    for f in range(folds):
        t = 0
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        for load in loader[f]:
            G = []
            # construct graph
            for p1 in range(np.array(load.edge_index).shape[1]):
                G.append((int(load.edge_index[0][p1]),int(load.edge_index[1][p1])))
            # calculate graph properties
            constant = G_property(G, constant_bool=1)
            degrees, graph = G_property(G, degree_bool=1, bin_bool=0) 
            clustering, graph = G_property(G, clustering_bool=1, bin_bool=0) 
            pagerank, graph = G_property(G, pagerank_bool=1, bin_bool=0)
            avg_path_len_G, graph = G_property(G, avg_path_length_bool=1, bin_bool=0)

            matrix = torch.cat((constant,degrees),1)
            matrix = torch.cat((matrix,clustering),1)
            matrix = torch.cat((matrix,pagerank),1)
            matrix = torch.cat((matrix,avg_path_len_G),1)

            matrix = matrix.numpy()
            matrix = pd.DataFrame(matrix,columns = ['Constant_feature','Degree','Clustering_coefficient','Pagerank','Aver_path_len'])   
            name = '/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/{}/{}_property{}{}_fold{}.txt'.format(dn, dn, t, task, f)
            matrix.to_csv(name, sep = '\t', index=False)
            t+=1

def train(i, dn, model, task, optimizer, train_loader, device, folds):
    ### for example, the folds that need to be trained are 0-8, the valid fold then is 9
    ### if total number of folds is equal to 10 
    model.train()
    correct_arr = []
    tot_loss = []
    length = []
    for f in folds:
        t= 0
        correct = 0
        total_loss = 0
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)

        for load in train_loader[f]:

            name = '/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/{}/{}_property{}{}_fold{}.txt'.format(dn, dn, t, task, f)
            property_file = pd.read_csv(name, sep = '\t')
            propert_i = property_file.iloc[:,list(i)] if isinstance(i,tuple) else property_file.iloc[:,[i]]
            array = np.array(propert_i)
            load.x = torch.cat((load.x, torch.tensor(array).float()), dim = 1)

            load = load.to(device)
            optimizer.zero_grad()
            out = model(load, load.batch)

            loss = F.nll_loss(out,load.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(load.y)
            with torch.no_grad():
                load = load.to(device)
                pred = model(load, load.batch).max(dim=1)[1]
            correct += pred.eq(load.y).sum().item()
            t+=1
        correct_arr.append(correct)
        length.append(len(train_loader[f].dataset))
        tot_loss.append(total_loss)
    
    return sum(correct_arr)/sum(length), sum(tot_loss)/sum(length)

def valid(i, dn, model, task, train_loader, device, fold):  
    model.eval()

    correct = 0
    t = 0
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    for load in train_loader[fold]:
        name = '/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/{}/{}_property{}{}_fold{}.txt'.format(dn, dn, t, task, fold)
        property_file = pd.read_csv(name, sep = '\t')

        propert_i = property_file.iloc[:,list(i)] if isinstance(i,tuple) else property_file.iloc[:,[i]]
        array = np.array(propert_i)
        load.x = torch.cat((load.x, torch.tensor(array).float()), dim = 1)
        
        with torch.no_grad():
            load = load.to(device)
            pred = model(load, load.batch).max(dim=1)[1]
        correct += pred.eq(load.y).sum().item()
        t+=1
    valid_acc = correct / len(train_loader[fold].dataset)
    return valid_acc

def test(i, dn,  model, task, test_loader, device, fold):
    correct = 0
    model.eval()
    total_num_nodes = 0
    t = 0
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    for load in test_loader[fold]:
        name = '/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/{}/{}_property{}{}_fold{}.txt'.format(dn, dn, t, task, fold)
        property_file = pd.read_csv(name, sep = '\t')
        propert_i = property_file.iloc[:,list(i)] if isinstance(i,tuple) else property_file.iloc[:,[i]]
        array = np.array(propert_i)
        load.x = torch.cat((load.x, torch.tensor(array).float()), dim = 1)
        
        with torch.no_grad():
            load = load.to(device)
            pred = model(load, load.batch).max(dim=1)[1]
        correct += pred.eq(load.y).sum().item()
        t+=1
    test_acc = correct / len(test_loader[fold].dataset)
    return test_acc


if __name__ == '__main__':
    o = option()
    o.multiple_dataset = True
    folds = 10 # k-fold cross-validation
    saved = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = []
    datasets.append(o.dataset)
    plt.figure()
    for dataset in datasets:
        ans = all_possible_concatenation(o)
        min_ans_len, max_ans_len = max_len_arr(ans)
        c_index = 0
        path = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
        data_set = TUDataset(root = path + o.dataset, name = o.dataset, use_node_attr = False)

        num_train_graphs = int(len(data_set) * 0.8)
        num_test_graphs = int(len(data_set) * 0.2)
        # fix shuffle seeds
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        perm = torch.randperm(len(data_set))
        train_idx = perm[:num_train_graphs]
        test_idx = perm[num_train_graphs:]
        num_each_fold = int(num_train_graphs / folds)
        batchsize = 16 if dataset != 'NCI1' else 32
        train_loader = []
        test_loader = []
        test_loader.append(DataLoader(data_set[test_idx], batch_size = batchsize , shuffle = False)) #### batch size 32 for NCI1
        # split cross-validation sets
        for i in range(folds):
            # fix shuffle seeds
            random.seed(1)
            torch.manual_seed(1)
            torch.cuda.manual_seed(1)
            np.random.seed(1)
            train_loader.append(DataLoader(data_set[train_idx[(i*num_each_fold):((i+1)*num_each_fold)]], batch_size = batchsize , shuffle=True, worker_init_fn=random.seed(12345))) 
        
        input_dim = 0
        first = 0
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        ans = [(3,4)]
        for i in train_loader[1:]:
            for load in i:
                input_dim = load.x.shape[1]
                first = load.x.shape[0]
                break
        input_dim+= len(ans[0])
        print(input_dim)
        #print(first)
        
        num_classes = {'ENZYMES':6, 'PROTEINS':2, 'NCI1':2}
        # reserve train_loader and test_loader
        '''
        reserve('train', dataset, train_loader, folds)
        
        reserve('test', dataset, test_loader, 1)
        '''
        
        mean_acc = [[] for i in range(min_ans_len, max_ans_len+1)]
        ans = [(3,4)]
        folds_arr = [i for i in range(folds)]
        folds_arr = np.array(folds_arr)
        for value in ans: # for each combination entry:
            model =  StrucFeaGNN(concat_fea_num = 2, embed_method = 'GIN', input_dim = input_dim, output_dim = num_classes[o.dataset], depth = 3).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.017, weight_decay=1e-5)
            best_epoch = 0
            best_train_acc = 0
            best_valid_acc = 0
            best_test_acc = 0
            op_iters = 0
            for epoch in range(1, 800):
                if o.dataset == 'NCI1':
                    if o.aim_feature == 2:
                        break
                # for train
                for fo in range(folds):
                    tr_acc, t_loss = train(value, o.dataset, model, 'train', optimizer, train_loader, device, folds_arr[folds_arr!=fo])
                    # for valid 
                    v_acc = valid(value, o.dataset, model, 'train',  train_loader, device, fo)
                    # for test
                    t_acc = test(value, o.dataset, model, 'test', test_loader, device, 0)


                    if v_acc > best_valid_acc:
                        best_valid_acc = v_acc
                        best_test_acc = t_acc
                        best_train_acc = tr_acc
                        best_epoch = epoch
                        #torch.save(model, '/home/jiaqing/桌面/Fea2Fea/src/model_pkl/best_model_{}.pkl'.format(o.dataset))
                        op_iters=0
                    op_iters+=1
                    if op_iters > 100:
                        break
                    print('Epoch {:03d}, Train loss:{:.4f}, best Train acc: {:.4f}, best valid acc :{:.4f}, best test acc : {:.4f}'.format(
   epoch, t_loss, best_train_acc, best_valid_acc , best_test_acc))
                    #model2 = torch.load('/home/jiaqing/桌面/Fea2Fea/src/model_pkl/best_model_{}.pkl'.format(o.dataset))
                    #print_conf_mtx(o.dataset, 'test', model2, test_loader, device, num_classes[o.dataset])
                #mean_acc[len(value) - min_ans_len].append(best_test_acc)
            break
            
        
        mean_acc_ = [ sum(mean_acc[i])/len(mean_acc[i]) for i in range(len(mean_acc))]
        mean_acc_ = [float('{:.4f}'.format(i)) for i in mean_acc_] 
        std_acc = [statistics.stdev(mean_acc[i]) for i in range(len(mean_acc))]
        saved.append(mean_acc_)
        saved.append(std_acc)
        
        x_axis = [i for i in range(min_ans_len, max_ans_len+1)]

        c_index+=1
        break

    # save mean acc and std acc   
    saved = pd.DataFrame(saved)
    