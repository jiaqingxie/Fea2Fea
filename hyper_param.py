from argparse import ArgumentParser
from model.GNN import Net
import numpy as np 
import pandas as pd 

def option():
    parser = ArgumentParser()
    parser.add_argument('--input_feature', default = 1, type = int, help = 'input feature')
    parser.add_argument('--output_feature', default = 2, type = int, help = 'output feature')
    parser.add_argument('--task', default = 'node', type = str, help = 'node / graph dataset')
    parser.add_argument('--dataset', default = 'Cora', type = str, help = 'dataset name')
    parser.add_argument('--hyperparameter', default = 'binning', type = str, help = 'hyper-para task')
    parser.add_argument('--min_bins', default = 2, type = int, help = 'minimum number of bins')
    parser.add_argument('--max_bins', default = 6, type = int, help = 'maximum number of bins')
    parser.add_argument('--min_depth', default = 2, type = int, help = 'minimum depth of GNN architecture')
    parser.add_argument('--max_depth', default = 6, type = int, help = 'maximum depth of GNN architecture')
    return parser.parse_args()

if __name__ == "__main__":
    a = option() # add parser
    
    if a.task == 'node':

    elif a.task == 'graph':
        

