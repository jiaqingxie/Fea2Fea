import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import torch

def G_property(graph,constant_bool = 0, degree_bool = 0, clustering_bool = 0, pagerank_bool = 0, 
                avg_path_length_bool = 0, bin_bool = 0):
    G=nx.Graph()
    G.add_edges_from(graph) #one example from CS224W
    # to reorder the mixed dictionary to an ascending order
    if constant_bool:
        constant = torch.ones([max(G) + 1,1],dtype = float)
        return constant

    if degree_bool:
        degrees = torch.zeros([max(G) + 1,1],dtype = float)
        for (n, d) in sorted(G.degree()):
            degrees[n,0] = d
        #print(degrees)
        
        degrees_sequence = sorted(degrees, reverse=True)
        '''
        plt.loglog(degrees_sequence, marker="o")
        plt.title("Degree distribution of Cora dataset")
        plt.ylabel("Degree")
        plt.xlabel("Numbers of node")
        plt.show()
        '''
        if bin_bool:
            bin_degrees = binning(degrees, k = 6)
            return bin_degrees, degrees, G
        else:
            return degrees,G

    if clustering_bool:
        clustering = torch.zeros([max(G) + 1,1],dtype = float)
        for key, value in sorted(nx.clustering(G).items()):
            clustering[int(key),0] = value
        clustering_sequence = sorted(clustering, reverse=True)
        if bin_bool:
            bin_clustering = binning(clustering, k = 6)
            return bin_clustering, clustering, G
        else:
            return clustering,G

    if pagerank_bool:
        pagerank = torch.zeros([max(G) + 1,1],dtype = float)
        for key,value in sorted(nx.pagerank(G, alpha=0.9).items()): #default dumping coefficient is equal to 0.85
            pagerank[key,0] = value
        pagerank_sequence = sorted(pagerank, reverse=True)
        if bin_bool:
            bin_pagerank = binning(pagerank, k = 6)
            return bin_pagerank, pagerank, G
        else:
            return pagerank,G
    
    if avg_path_length_bool:
        avg_path_len_G = torch.zeros([max(G) + 1,1],dtype = float)
        for i in range(max(G) + 1):
            p = torch.tensor([],dtype = float)
            for j in range(max(G) +1):
                if i !=j: 
                    try:    
                        k = torch.tensor([nx.shortest_path_length(G,source = i, target = j)],dtype = float)
                        p = torch.cat((p,k),0)
                    except:
                        pass
            if p.nelement() == 0:
                k = 0
            else:
                k = sum(p) / len(p)
                k = k.item()
            
            avg_path_len_G[i] = k
            #print(i)
        #avg_path_len_G_sequence = sorted(avg_path_len_G, reverse=True)
        return avg_path_len_G,G
 



    if centrality_bool:  #degree centrality of nodes
        pass

def binning(Array, k, data_len):
    Array = Array.reshape(data_len,)
    Array = Array.tolist()
    if Array.count(min(Array)) <= 1/5 * len(Array):
        a = np.array(sorted(Array))
        #print(a)
        bins = []
        for i in range(k-1):
            bins.append(a[int(i*math.floor(len(Array)/k)+math.floor(len(Array)/k))])
        
        Array = np.digitize(Array,bins,right = True)
        Array = torch.tensor(Array).long()
        return Array
    else:
        temp = Array[:]
        #print(len(temp))
        S = min(Array)
        minimum = min(Array)
        while minimum in temp:
            temp.remove(minimum)
        a = np.array(sorted(temp))
        bins = []
        for i in range(k-2):
            bins.append(a[int(i*math.floor(len(temp)/(k-1))+math.floor(len(temp)/(k-1)))])
        temp = np.digitize(temp, bins, right = True)
        temp = temp.tolist()
        for i in range(len(Array)):
            if Array[i] == S:
                Array[i] = 0
            else:
                Array[i] = temp[0] + 1
                temp.remove(temp[0])
        Array = torch.tensor(Array).long()
        return Array
        



