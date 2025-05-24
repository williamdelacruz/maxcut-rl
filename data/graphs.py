import networkx as nx
import numpy as np

def generate_random_graph(n, p):
    G = nx.erdos_renyi_graph(n=n, p=p)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.rand()
    return G
