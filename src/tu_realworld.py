import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
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
from model.StrucFea_GNN import StrucFeaGNN
from utils import max_len_arr
from f_f_TU import train, valid, test


if __name__ == '__main__':
    o = option()
    o.multiple_dataset = True
    saved = []
    '''
    this program aims at a specific feature, if you want to get a
    threshold impact graph, find the python notebook under this folder

    '''
    #print(ans) will show all possible concatenation under one threshold
    # ans = [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (2, 3, 4), (0, 2, 3, 4)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if o.multiple_dataset:
        datasets = ['ENZYMES', 'PROTEINS', 'NCI1']
        #c = ['r', 'b', 'g']
    else:
        datasets = o.dataset
        #c = ['b']
    plt.figure()

    for dataset in datasets:
        o.dataset = dataset
        ans = all_possible_concatenation(o)
        min_ans_len, max_ans_len = max_len_arr(ans)
        c_index = 0
        path = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
        data_set = TUDataset(root = path + dataset, name = dataset, use_node_attr = False)

        train_len, valid_len= int(0.8 * len(data_set)), int(0.1 * len(data_set))  
        test_len = len(data_set) - train_len - valid_len
        batchsize = 16 if dataset != 'NCI1' else 32
        train_loader = DataLoader(data_set[0:train_len], batch_size = batchsize , shuffle=False) #### batch size 32 for NCI1
        valid_loader = DataLoader(data_set[train_len:(train_len+valid_len)], batch_size = batchsize , shuffle = False) #### batch size 32 for NCI1
        test_loader = DataLoader(data_set[(train_len+valid_len):len(data_set)], batch_size = batchsize , shuffle = False) #### batch size 32 for NCI1


        mean_acc = [[] for i in range(min_ans_len, max_ans_len+1)]

        for value in ans: # for each combination entry:
            # should transform value to list 
            for i in range(10):
                model =  augGNN(input_dim = len(value)).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.015, weight_decay=1e-4)

                best_epoch = 0
                best_valid_acc = 0
                best_test_acc = 0
                op_iters = 0

                for epoch in range(1, 3000):
                    if dataset == 'NCI1':
                        if o.aim_feature == 2:
                            break
                    # for train
                    t_loss = train(value, o.aim_feature, dataset, model, 'train', optimizer, train_loader, device)
                    # for valid 
                    v_acc = valid(value, o.aim_feature, dataset, model, 'valid', optimizer, valid_loader, device)
                    # for test
                    t_acc = test(value, o.aim_feature, dataset, model, 'test', optimizer, test_loader, device)
                    print('Epoch {:03d}, Train Loss: {:.4f}, Valid acc :{:.4f}, Test acc : {:.4f}'.format(
                       epoch, t_loss, v_acc, t_acc ))

                    if v_acc > best_valid_acc:
                        best_valid_acc = v_acc
                        best_test_acc = t_acc
                        best_epoch = epoch
                        

                        op_iters=0
                    op_iters+=1
                    if op_iters > 20:
                        break

                mean_acc[len(value) - min_ans_len].append(best_test_acc)
            
        
        mean_acc_ = [ sum(mean_acc[i])/len(mean_acc[i]) for i in range(len(mean_acc))]
        mean_acc_ = [float('{:.4f}'.format(i)) for i in mean_acc_] 
        std_acc = [statistics.stdev(mean_acc[i]) for i in range(len(mean_acc))]
        saved.append(mean_acc_)
        saved.append(std_acc)
        
        x_axis = [i for i in range(min_ans_len, max_ans_len+1)]

        #plt.plot(x_axis, mean_acc_, )
        plt.errorbar(x_axis, mean_acc_ , yerr = std_acc, fmt='o-', elinewidth=2,capsize=4,label=dataset)
        #ax.grid(alpha=0.5, linestyle=':')
        c_index+=1

    # save mean acc and std acc   
    saved = pd.DataFrame(saved)
    path =  '/home/jiaqing/桌面/Fea2Fea/Result/tud_acc_record_'+ str(o.aim_feature) +'.txt'
    saved.to_csv(path, header = None, index = None, sep = '\t')    
    plt.xlabel("Number of input graph features")
    plt.ylabel("Mean test accuracy")
    t_label =  ['Constant','Degree','Clustering','PageRank','Aver_Path_Len']
    plt.title("aim feature: {}, threshold: {}".format(t_label[o.aim_feature], o.threshold))
    plt.legend(loc = 'lower right')
    path = '/home/jiaqing/桌面/Fea2Fea/Result/pipeline/'
    plt.savefig(path + 'tud_aim_' + str(o.aim_feature) + '.eps', dpi = 400, bbox_inches='tight')
    plt.show()
