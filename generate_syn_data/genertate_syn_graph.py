import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from ... import G_property
import networkx as nx


sys.path.insert(1, '/path/to/application/app/folder')
'''
G = nx.random_geometric_graph(200, 0.125)
nx.draw(G)
plt.draw()#
plt.show()
'''

def generate_property(edge_idx):
    G = []
    edge_idx = np.array(edge_idx)
    for u,v in edge_idx:
        G.append((u,v))

    


def option():
    parser = ArgumentParser()
    parser.add_argument('--num_nodes', default=400, type=int, help='number of nodes')
    parser.add_argument('--threshold', default=0.125, type=float, help='distance threshold value')
    return parser.parse_args()

if __name__ == "__main__":
    o = option()
    # find if we have stored a random geometric graph already
    edge_idx_file = 'geometric_graph_{}_edge_idx.txt'.format(o.num_nodes)
    #node_attr_file = 'geometric_graph_{}_node_attr.txt'.format(o.num_nodes)
    edge_idx_file = '/home/jiaqing/桌面/Fea2Fea/data/syn_data/' + edge_idx_file
    if os.path.exists(edge_idx_file):
        print("A geometric graph with {} nodes has already been generated".format(o.num_nodes))
    # if synthetic geometric graph is not generated
    else:
        G = nx.random_geometric_graph(o.num_nodes, o.threshold)
        pos = nx.get_node_attributes(G, "pos")

        # find node near center (0.5,0.5)
        dmin = 1
        ncenter = 0
        for n in pos:
            x, y = pos[n]
            d = (x - 0.5) ** 2 + (y - 0.5) ** 2
            if d < dmin:
                ncenter = n
                dmin = d

        # color by path length from node near center
        p = dict(nx.single_source_shortest_path_length(G, ncenter))

        plt.figure(figsize=(12, 12))
        nx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4)
        nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(p.keys()),
        node_size=80,
        node_color=list(p.values()),
        cmap=plt.cm.Reds_r,
        )
        # save visualization to path
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.title("geometric graph with node num {}".format(o.num_nodes),y=-0.1,fontdict={'weight':'normal','size': 20})
        plt.axis("off")
        plt.savefig('/home/jiaqing/桌面/Fea2Fea/generate_syn_data/graph_{}.eps'.format(o.num_nodes), dpi=400)
        #plt.show()
        row = 0

        # save to edge index file
        tmp = pd.DataFrame(columns=['u', 'v'])
        for u,v in G.edges():
            # 1. add both (u,v) and (v, u)
            tmp.loc[row] = [u,v]
            row+=1
            tmp.loc[row] = [v,u]
            row+=1
            # 2. ordering
        tmp = tmp.sort_values(by='u')
        tmp.to_csv(edge_idx_file, header = None, index = None, sep = ',')  



    
    