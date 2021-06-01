from graph_property import G_property
import numpy as np
import pandas as pd 
import torch
import torch_geometric
from torch_geometric.datasets import TUDataset


#Bioinformatics, social networks, small molecules
tuda = ['ENZYMES','NCI1','PROTEINS']
#,'COLLAB',,'REDDIT-MULTI-5K','ZINC_test'

for name_1 in tuda:

    dataset = TUDataset(root='/tmp/' + name_1, name=name_1, use_node_attr=False)
    cnt = 0
    for data in dataset:
    #print(data.edge_index)
        G = []
        for i in range(np.array(data.edge_index).shape[1]):
            G.append((int(data.edge_index[0][i]),int(data.edge_index[1][i])))
       
        constant = torch.ones([len(data.x),1], dtype = float)
        degrees, graph = G_property(G, degree_bool=1, bin_bool=0)
        #print(data.x.shape)
        #print(degrees.shape)
        clustering, graph = G_property(G, clustering_bool=1, bin_bool=0) 
        #print(clustering.shape)
        pagerank, graph = G_property(G, pagerank_bool=1, bin_bool=0)
        avg_path_len_G, graph = G_property(G, avg_path_length_bool=1, bin_bool=0)
        
        if(data.x.shape[0] == degrees.shape[0]): #-------- TEST ------ -- # 
            matrix = torch.cat((constant,degrees),1)
            matrix = torch.cat((matrix,clustering),1)
            matrix = torch.cat((matrix,pagerank),1)
            matrix = torch.cat((matrix,avg_path_len_G),1)
            matrix = matrix.numpy()
            matrix = pd.DataFrame(matrix,columns = ['Constant_feature','Degree','Clustering_coefficient','Pagerank','Aver_path_len'])
            name = r'/home/jiaqing/桌面/Fea2Fea/Result/TUdataset/'+ name_1 +'/' + name_1 + '_property' + str(cnt) + '.txt'
            matrix.to_csv(name, sep = '\t', index=False)
            cnt+=1
        else:
            continue
        
