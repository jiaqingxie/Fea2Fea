# traverse all
import numpy as np 
import pandas as pd
from argparse import ArgumentParser

def option():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='Cora', type=str, help='dataset')
    return parser.parse_args()

def get_optimal_R():
    '''
    return the optimal R for the given task and correspomding graph embedding method   
    '''
    o = option()
    
    graph_embed  = ['SAGE','GAT','GCN','GIN']
    planetoid = ['Cora', 'Citeseer', 'PubMed']
    tudataset = ['ENZYMES','PROTEINS', 'NCI1']

    optimal_R = [[0] * 5 for i in range(0, 5)]
    optimal_method = [[0] * 5 for i in range(0, 5)]

    if o.dataset in planetoid:
        path = '/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/' # add your path here
    elif o.dataset in tudataset:
        path = '/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/' # add your path here
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
    fig_name1 = o.dataset + '_optimal_R.txt'
    fig_name2 = o.dataset + '_optimal_method.txt'
    fig_path1 = path + fig_name1
    fig_path2 = path + fig_name2
    optimal_R.to_csv(fig_path1, header = None, index = None, sep = '\t')
    optimal_method.to_csv(fig_path2, header = None, index = None, sep = '\t')
                    
def all_possible_concatenation():
    


if __name__ == '__main__':

    get_optimal_R()
    



