from graph_property import G_property
import numpy as np
import pandas as pd 
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch



path = osp.join( '/home/jiaqing/桌面/FASG_KDD/data') #桌面 change to Desktop under English version


# -------------------- overall 16-18 hours ----------------------------#

planetoid = ['Cora','Citeseer','PubMed']

for dataset in planetoid:
    #dataset = Planetoid(path, dataset, transform=T.TargetIndegree())
    dataset_name = dataset
    dataset = Planetoid(path, name = dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    G = []
    print(data.edge_index)
    for i in range(np.array(data.edge_index).shape[1]):
        G.append((int(data.edge_index[0][i]),int(data.edge_index[1][i])))

    constant = torch.ones([len(data.x),1], dtype = float)
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
    name = r'/home/jiaqing/桌面/Fea2Fea/Result/Planetoid/' + dataset_name + '_property.txt'
    matrix.to_csv(name, sep = '\t', index=False)



