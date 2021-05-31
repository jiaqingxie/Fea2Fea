from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os.path as osp

import torch 
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import Dataset, DataLoader

from model.var_GNN import Net
from model.aug_GNN import augGNN
from optimal_R import option, all_possible_concatenation
from graph_property import G_property, binning
from utils import max_len_arr

from graph_property import binning, G_property
from f_f_TU import train, valid, test

def train_n(task):
    '''
        abbreviation of training on node dataset
        avoid train from f_f_TU
    '''
    if task == 'node':
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
        #print(loss)
        loss.backward()
        
        optimizer.step()
        
    elif task == 'graph':
        pass

def test_n(task):
    '''
        abbreviation of testing on node dataset
        avoid test from f_f_Tu
    '''
    if task == 'node':
        model.eval()
        logits, accs = model(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs

def option():
    parser = ArgumentParser()
    parser.add_argument('--input_feature', default = 0, type = int, help = 'input feature')
    parser.add_argument('--aim_feature', default = 1, type = int, help = 'output feature')
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
    parser.add_argument('--threshold', default=0.8, type=float, help='threshold')
    return parser.parse_args()

if __name__ == "__main__":
    a = option() # add parser
    path = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = ['Constant','Degree','Clustering','PageRank','Aver_Path_Len']
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
        propert_j = property_file.iloc[:,[a.aim_feature]]
        array_2 = np.array(propert_j)
        # if task is different bins
        print("-------- start testing --------")
        if a.hyperparameter == 'binning':
            print("dataset : {}".format(a.dataset))
            print("dataset type : {}".format(a.task))
            print("min bins : {}".format(a.min_bins))
            print("max bins : {}".format(a.max_bins))
            print("input fearure : {}".format(features[a.input_feature]))
            print("aim fearure : {}".format(features[a.aim_feature]))
            print("graph embedding method : {}".format(a.embedding))
            average = 10
            for bins in range(a.min_bins, a.max_bins +1):
                # test each case for 10 times
                test_acc_arr = []
                avg_test_acc = 0
                for avg in range(average):
                    best_val_acc = test_acc = 0
                    t = 0

                    data.y = binning(array_2, k = bins, data_len =  len(data.y))
                    # to GPU
                    model = Net(embedding = a.embedding ,bins = bins).to(device)
                    data =  data.to(device)
                    # optimizer
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)    
                    # training epoch
                    for epoch in range(1, 3000):
                        train_n('node')
                        train_acc, val_acc, tmp_test_acc = test_n('node')
                        if val_acc > best_val_acc and tmp_test_acc > test_acc:
                            best_val_acc = val_acc
                            test_acc = tmp_test_acc
                            t = 0
                        t = t + 1
                        if t > 500:
                            break   
                    # calculate average accuracy
                    test_acc_arr.append(test_acc)
                avg_test_acc = sum(test_acc_arr)/len(test_acc_arr)
                avg_test_acc = round(avg_test_acc, 3)
                log2 = 'bins : {}, test acc : {:.3f}, std: {:.3f} task: input {} predict output {}'
                print(log2.format(bins, avg_test_acc, np.std(test_acc_arr, ddof=1), features[a.input_feature], features[a.aim_feature]))

                    
        elif a.hyperparameter == 'depth':
            average = 10
            #  draw bars 
            plt_acc = [] # mean accuracy
            plt_dep = []
            plt_all = []
            plt_std = []
            for dep in range(a.min_depth, a.max_depth + 1, 2):
                avg_test_acc = 0
                plt_dep.append(dep)
                test_acc_arr = []
                for avg in range(average):
                    best_val_acc = test_acc = 0
                    t = 0
                    # choose k = 6
                    data.y = binning(array_2, k = 6, data_len =  len(data.y))
                    # to GPU
                    model = Net(embedding = a.embedding , depth = dep, bins = 6).to(device)
                    data =  data.to(device)
                    # optimizer
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.04, weight_decay=5e-4)    
                    # training epoch
                    for epoch in range(1, 3000):
                        train_n('node')
                        train_acc, val_acc, tmp_test_acc = test_n('node')
                        if val_acc > best_val_acc and tmp_test_acc > test_acc:
                            best_val_acc = val_acc
                            test_acc = tmp_test_acc
                            t = 0
                        t = t + 1
                        if t > 500:
                            break   
                    # calculate average accuracy
                    avg_test_acc+= test_acc
                    plt_all.append(test_acc)
                    test_acc_arr.append(test_acc)
                plt_std.append(np.std(plt_all))
                avg_test_acc = sum(test_acc_arr)/len(test_acc_arr)
                avg_test_acc = round(avg_test_acc, 3)
                plt_acc.append(avg_test_acc)
                log2 = 'depth : {}, test acc : {:.3f}, std : {:.3f}, task: input {} predict output {}'
                print(log2.format(dep, avg_test_acc, np.std(test_acc_arr, ddof=1), features[a.input_feature], features[a.aim_feature]))

            # ----- line plot ----- #
            
            plt.figure(figsize=(8,8))
            plt_dep = np.array(plt_dep)
            plt_acc = np.array(plt_acc)
            plt_std = np.array(plt_std)

            plt.scatter(plt_dep,plt_acc)
            plt.plot(plt_dep, plt_acc)
            plt.fill_between(plt_dep, plt_acc - plt_std, plt_acc + plt_std, facecolor='green', alpha=0.3)
            
            plt.xlabel('depth')
            plt.ylabel('avr_acc')
            plt.savefig('depth_results.eps', dpi = 800, format = 'eps')
            plt.show()
            
        elif a.hyperparameter == 'threshold':
            threshold = [0.6, 0.8]
            

            print("dataset : {}".format(a.dataset))
            print("dataset type : {}".format(a.task))
            print("aim fearure : {}".format(features[a.aim_feature]))
            print("graph embedding method : {}".format(a.embedding))
            average = 10
            for thres in threshold:
                a.threshold = thres
                ans = all_possible_concatenation(a)
                min_ans_len, max_ans_len = max_len_arr(ans)
                for value in ans:
                    data.x = np.array(property_file.iloc[:,list(value)])
                    #print(data.x.shape)
                    data.x = torch.tensor(data.x).float()
                    data =  data.to(device)
                    data.y = binning(array_2, k = 6, data_len =  len(data.y))
                        # to GPU
                    data =  data.to(device)
                    # test each case for 10 times
                    test_acc_arr = []
                    avg_test_acc = 0
                    for avg in range(average):
                        best_val_acc = test_acc = 0
                        t = 0
                        # optimizer
                        model =  augGNN(input_dim = len(value)).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.015, weight_decay=1e-4)  
                        # training epoch
                        for epoch in range(1, 3000):
                            train_n('node')
                            train_acc, val_acc, tmp_test_acc = test_n('node')
                            if val_acc > best_val_acc and tmp_test_acc > test_acc:
                                best_val_acc = val_acc
                                test_acc = tmp_test_acc
                                t = 0
                            t = t + 1
                            if t > 500:
                                break   
                        # calculate average accuracy
                        test_acc_arr.append(test_acc)
                    avg_test_acc = sum(test_acc_arr)/len(test_acc_arr)
                    avg_test_acc = round(avg_test_acc, 3)
                log2 = 'threshold : {}, test acc : {:.3f}, std: {:.3f}  predict output {}'
                print(log2.format(thres, avg_test_acc, np.std(test_acc_arr, ddof=1), features[a.aim_feature]))

    
    # else if graph dataset
    elif a.task == 'graph':
        dataset = TUDataset(root = '/home/jiaqing/桌面/Fea2Fea/data/' + a.dataset, name = a.dataset, use_node_attr = False)
        # data loader
        train_len, valid_len= int(0.8 * len(dataset)), int(0.1 * len(dataset))
        test_len = len(dataset) - train_len - valid_len
        # !!
        # you should change the batch size to 32 if you want to have tests on NCI1 dataset.
        train_loader = DataLoader(dataset[0:train_len], batch_size = 16, shuffle=False)
        valid_loader = DataLoader(dataset[train_len:(train_len+valid_len)], batch_size = 16, shuffle = False)
        test_loader = DataLoader(dataset[(train_len+valid_len):len(dataset)], batch_size = 16, shuffle = False)

        print("-------- start testing --------")
        if a.hyperparameter == 'binning':
            print("dataset : {}".format(a.dataset))
            print("dataset type : {}".format(a.task))
            print("min bins : {}".format(a.min_bins))
            print("max bins : {}".format(a.max_bins))
            print("input fearure : {}".format(features[a.input_feature]))
            print("aim fearure : {}".format(features[a.aim_feature]))
            print("graph embedding method : {}".format(a.embedding))
            average = 10
            for bins in range(a.min_bins, a.max_bins +1):
                # test each case for 10 times
                test_acc_arr = []
                avg_test_acc = 0
                for avg in range(average):
                    best_val_acc = test_acc = 0
                    t = 0
                    model = Net(embedding = a.embedding ,bins = bins).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr = 0.04)
                    for epoch in range(1, 200):
                        # for train
                        t_loss = train(a.input_feature, a.aim_feature, a.dataset, model, 'train', optimizer, train_loader, device, k = bins)
                        # for valid 
                        v_acc = valid(a.input_feature, a.aim_feature, a.dataset, model, 'valid', optimizer, valid_loader, device,  k = bins)
                        # for test
                        t_acc = test(a.input_feature, a.aim_feature, a.dataset, model, 'test', optimizer, test_loader, device, k = bins)
                        if v_acc > best_val_acc:
                            best_val_acc = v_acc
                            test_acc = t_acc
                            best_epoch = epoch
                            op_iters=0
                        op_iters+=1
                        if op_iters > 20:
                            break
                    test_acc_arr.append(test_acc)
                avg_test_acc = sum(test_acc_arr) / len(test_acc_arr)
                avg_test_acc = round(avg_test_acc, 3)
                log2 = 'bins : {}, test acc : {:.3f}, std: {:.3f}, task: input {} predict output {}'
                print(log2.format(bins, avg_test_acc, np.std(test_acc_arr, ddof=1), features[a.input_feature], features[a.aim_feature]))
                    
        elif a.hyperparameter == 'depth':
            print("dataset : {}".format(a.dataset))
            print("dataset type : {}".format(a.task))
            print("min depth : {}".format(a.min_depth))
            print("max depth : {}".format(a.max_depth))
            print("input fearure : {}".format(features[a.input_feature]))
            print("aim fearure : {}".format(features[a.aim_feature]))
            print("graph embedding method : {}".format(a.embedding))
            average = 10
            for dep in range(a.min_depth, a.max_depth + 1, 2):
                # test each case for 10 times
                test_acc_arr = []
                avg_test_acc = 0
                for avg in range(average):
                    best_val_acc = test_acc = 0
                    t = 0
                    model = Net(embedding = a.embedding , depth = dep, bins = 6).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr = 0.04)
                    for epoch in range(1, 200):
                        # for train
                        t_loss = train(a.input_feature, a.aim_feature, a.dataset, model, 'train', optimizer, train_loader, device, k = 6)
                        # for valid 
                        v_acc = valid(a.input_feature, a.aim_feature, a.dataset, model, 'valid', optimizer, valid_loader, device,  k = 6)
                        # for test
                        t_acc = test(a.input_feature, a.aim_feature, a.dataset, model, 'test', optimizer, test_loader, device, k = 6)
                        if v_acc > best_val_acc:
                            best_val_acc = v_acc
                            test_acc = t_acc
                            best_epoch = epoch
                            op_iters=0
                        op_iters+=1
                        if op_iters > 20:
                            break
                    test_acc_arr.append(test_acc)
                avg_test_acc = sum(test_acc_arr) / len(test_acc_arr)
                avg_test_acc = round(avg_test_acc, 3)
                log2 = 'depth : {}, test acc : {:.3f}, std: {:.3f}, task: input {} predict output {}'
                print(log2.format(dep, avg_test_acc, np.std(test_acc_arr, ddof=1), features[a.input_feature], features[a.aim_feature]))
                    
        
        elif a.hyperparameter == 'threshold':
            print("dataset : {}".format(a.dataset))
            print("dataset type : {}".format(a.task))
            print("min depth : {}".format(a.min_depth))
            print("max depth : {}".format(a.max_depth))
            print("input fearure : {}".format(features[a.input_feature]))
            print("aim fearure : {}".format(features[a.aim_feature]))
            print("graph embedding method : {}".format(a.embedding))
            average = 10
            for dep in range(a.min_depth, a.max_depth + 1, 2):
                # test each case for 10 times
                test_acc_arr = []
                avg_test_acc = 0
                for avg in range(average):
                    best_val_acc = test_acc = 0
                    t = 0
                    model = Net(embedding = a.embedding , depth = dep, bins = 6).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr = 0.04)
                    for epoch in range(1, 200):
                        # for train
                        t_loss = train(a.input_feature, a.aim_feature, a.dataset, model, 'train', optimizer, train_loader, device, k = 6)
                        # for valid 
                        v_acc = valid(a.input_feature, a.aim_feature, a.dataset, model, 'valid', optimizer, valid_loader, device,  k = 6)
                        # for test
                        t_acc = test(a.input_feature, a.aim_feature, a.dataset, model, 'test', optimizer, test_loader, device, k = 6)
                        if v_acc > best_val_acc:
                            best_val_acc = v_acc
                            test_acc = t_acc
                            best_epoch = epoch
                            op_iters=0
                        op_iters+=1
                        if op_iters > 20:
                            break
                    test_acc_arr.append(test_acc)
                avg_test_acc = sum(test_acc_arr) / len(test_acc_arr)
                avg_test_acc = round(avg_test_acc, 3)
                log2 = 'depth : {}, test acc : {:.3f}, std: {:.3f}, task: input {} predict output {}'
                print(log2.format(dep, avg_test_acc, np.std(test_acc_arr, ddof=1), features[a.input_feature], features[a.aim_feature]))
                    
    