# traverse all
import numpy as np 
import pandas as pd
from argparse import ArgumentParser
from utils import powerset, subset
import os, sys

def option():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='Cora', type=str, help='dataset')
    parser.add_argument('--threshold', default=0.8, type=float, help='threshold')
    parser.add_argument('--task', default='getR', type=str, help='getR or getConcat')
    parser.add_argument('--aim_feature', default=1, type=int, help='graph features')
    return parser.parse_args()

def get_optimal_R(o):
    '''
    return the optimal R for the given task and correspomding graph embedding method   
    '''
    
    graph_embed  = ['SAGE','GAT','GCN','GIN']
    planetoid = ['Cora', 'Citeseer', 'PubMed']
    tudataset = ['ENZYMES','PROTEINS', 'NCI1']

    optimal_R = [[0] * 5 for i in range(0, 5)]
    optimal_method = [[0] * 5 for i in range(0, 5)]

    if o.dataset in planetoid:
        path = '/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/' # add your path of R here
    elif o.dataset in tudataset:
        path = '/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/' # add your path of R here
    else:
        print("please input a correct dataset name")
        sys.exit()
    for ge in graph_embed:

        df = pd.read_csv(path + o.dataset + '_' + ge +'.txt', sep = '\t',header=None) # a matrix
        
        rows = len(df.axes[0])
        cols = len(df.axes[1])

        for i in range(0, rows):
            for j in range(0, cols):
                if df.iloc[i,j] > optimal_R[i][j]:
                    optimal_R[i][j] = df.iloc[i,j]
                    optimal_method[i][j] = ge
    
    optimal_R = pd.DataFrame(optimal_R)
    optimal_method = pd.DataFrame(optimal_method)

    # obtain optimal R and corresponding graph embedding method
    txt_name1 = o.dataset + '_optimal_R.txt'
    txt_name2 = o.dataset + '_optimal_method.txt'
    txt_path1 = path + txt_name1
    txt_path2 = path + txt_name2
    optimal_R.to_csv(txt_path1, header = None, index = None, sep = '\t')
    optimal_method.to_csv(txt_path2, header = None, index = None, sep = '\t')
    return optimal_R, optimal_method
                    
def all_possible_concatenation(o):
    '''
    print all possible feature concatenation situations
    '''
    planetoid = ['Cora', 'Citeseer', 'PubMed']
    tudataset = ['ENZYMES','PROTEINS', 'NCI1']

    if o.dataset in planetoid:
        path = '/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/' 
    elif o.dataset in tudataset:
        path = '/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/' 
    else:
        print("please input a correct dataset name")
        sys.exit()
    txt_name = o.dataset + '_optimal_R.txt'
    txt_path = path + txt_name
    df = pd.read_csv(txt_path, sep = '\t',header=None)
    aim_feature = o.aim_feature
    rows = len(df.axes[0])
    cols = len(df.axes[1])
    ans = set()
    # alrorithm, traverse
    sets = [i for i in range(5)]
    #del sets[o.aim_feature]
    sets = list(powerset(sets))

    for row in range(0,rows):
        for col in range(0, cols):
            if df.iloc[row,col] < o.threshold and row!= o.aim_feature and row!= col:
                ans.add((row, col))
            else:
                pass
    
    print("all possible concatenations:")

    comb = []
    for i in range(2,5):
        print(str(i) + " features: ")
        tmp = []
        for j in range(len(sets)):
            if len(sets[j]) == i:
                ans2 = subset(sets[j], ans)
                if ans2 == True:
                    tmp.append(sets[j])
            else:
                continue
        print(tmp)
        for ele in tmp:
            comb.append(ele)
    return comb
        
if __name__ == '__main__':
    o = option()
    if o.task == 'getR':
        a, b = get_optimal_R(o)
    if o.task == 'getConcat':
        ans = all_possible_concatenation(o)
        print(ans)
