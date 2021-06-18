import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import statistics

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.data import DataLoader


from optimal_R import option, all_possible_concatenation
from graph_property import G_property, binning
from model.aug_GNN import augGNN
from utils import max_len_arr

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
    multiple_dataset = True
    saved = []
    '''
    this program aims at a specific feature, if you want to get a
    threshold impact graph, find the python notebook under this folder

    '''
    #print(ans) will show all possible concatenation under one threshold
    # ans = [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (2, 3, 4), (0, 2, 3, 4)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if multiple_dataset:
        datasets = ['Cora', 'PubMed', 'Citeseer']
        #c = ['r', 'b', 'g']
    else:
        datasets = o.dataset # set to defualt dataset
    plt.figure()
    
    for dataset in datasets:
        o.dataset = dataset
        ans = all_possible_concatenation(o)
        min_ans_len, max_ans_len = max_len_arr(ans)
        #saved = np.array([0 for i in range(min_ans_len, max_ans_len+1)])

        c_index = 0
        path = osp.join('/home/jiaqing/桌面/Fea2Fea/data/')
        data_set = Planetoid(path, name = dataset, transform=T.NormalizeFeatures())
        data = data_set[0]
        name = r'/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/' + dataset + '_property.txt'
        property_file = pd.read_csv(name, sep = '\t')

        
        data.y = np.array(property_file.iloc[:,[o.aim_feature]])
        #print(property_file.iloc[:,[1]])
        number = len(data.y)
        data.y = binning(data.y, k = 6,data_len =  number)

        mean_acc = [[] for i in range(min_ans_len, max_ans_len+1)]

        for value in ans: # for each combination entry:
            # should transform value to list 
            data.x = np.array(property_file.iloc[:,list(value)])
            #print(data.x.shape)
            data.x = torch.tensor(data.x).float()
            data =  data.to(device)
            for i in range(10):
                model =  augGNN(input_dim = len(value)).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.015, weight_decay=1e-4)
                t = 0
                best_val_acc = test_acc = 0 
                #train_accu = []
                #count = []
                for epoch in range(1, 3000):

                    train()
                    train_acc, val_acc, tmp_test_acc = test()
                    #if epoch%1 == 0:
                    #    count.append(epoch)
                    #    train_accu.append(train_acc)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        test_acc = tmp_test_acc
                        t = 0

                    t = t + 1
                    if t > 400:
                        break   
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    #print(log.format(epoch, train_acc, best_val_acc, test_acc))
                mean_acc[len(value) - min_ans_len].append(test_acc)
            
        
        mean_acc_ = [ sum(mean_acc[i])/len(mean_acc[i]) for i in range(len(mean_acc))]
        mean_acc_ = [float('{:.4f}'.format(i)) for i in mean_acc_] 
        std_acc = [statistics.stdev(mean_acc[i]) for i in range(len(mean_acc))]
        saved.append(mean_acc_)
        saved.append(std_acc)
        
        x_axis = [i for i in range(min_ans_len, max_ans_len+1)]

        #plt.plot(x_axis, mean_acc_, )np.array([0 for i in range(1, max_ans_len+1)])
        plt.errorbar(x_axis, mean_acc_ , yerr = std_acc, fmt='o-', elinewidth=2,capsize=4,label=dataset)
        #ax.grid(alpha=0.5, linestyle=':')
        c_index+=1

    # save mean acc and std acc   
    saved = pd.DataFrame(saved)
    path =  '/home/jiaqing/桌面/Fea2Fea/Result/acc_record_'+ str(o.aim_feature) +'.txt'
    saved.to_csv(path, header = None, index = None, sep = '\t')    
    plt.xlabel("Number of input graph features")
    plt.ylabel("Mean test accuracy")
    t_label =  ['Constant','Degree','Clustering','PageRank','Aver_Path_Len']
    plt.title("aim feature: {}, threshold: {}".format(t_label[o.aim_feature], o.threshold))
    plt.legend(loc = 'center')
    path = '/home/jiaqing/桌面/Fea2Fea/Result/pipeline/'
    plt.savefig(path + 'aim_' + str(o.aim_feature) + '.eps', dpi = 400, bbox_inches='tight')
    plt.show()
        