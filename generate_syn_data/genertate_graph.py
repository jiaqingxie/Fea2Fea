import networkx as nx
import os, sys
import matplotlib.pyplot as plt
G = nx.random_geometric_graph(200, 0.125)
nx.draw(G)
plt.draw()#
plt.show()


if __name__ == "__main__":
    # find if we have stored a random geometric graph already
    edge_inx_file = "a.txt"
    folder = os.path.join("")
    
    