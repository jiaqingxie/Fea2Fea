# traverse all
import numpy as np 
import pandas as pd
from argparse import ArgumentParser

def option():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='Cora', type=str, help='dataset')
    return parser.parse_args()

def get_optimal_R():
    o = option()
    '''
    return the optimal R for the given task and correspomding graph embedding method   
    '''
    
    graph_embed  = ['SAGE','GAT','GCN','GIN']
    planetoid = ['Cora', 'Citeseer', 'PubMed']
    tudataset = ['ENZYMES','PROTEINS', 'NCI1']

    for ge in graph_embed:
        if o.dataset in planetoid:
            path = '/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/'
        elif o.dataset in tudataset:
            path = '/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/'
        txt = pd.read_csv(path + o.dataset + '_' + ge +'.txt', sep = '\t',header=None) # a matrix
        print(txt)


if __name__ == '__main__':

    get_optimal_R()
    



