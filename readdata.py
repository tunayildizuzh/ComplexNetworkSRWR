import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

data = np.genfromtxt("/Users/tunayildiz/Desktop/UZH/ComplexNetworkSRWR/dataset/BTCAlphaNet.csv", delimiter=",", dtype=float)[0:50]
np.random.shuffle(data)


users = []
edges = []

for idx, (a,b,c) in enumerate(data):
    users.append(a)
    edges.append((a,b))
    if c < 0.4:
        data[idx][2] = -1
    if c > 0.4:
        data[idx][2] = 1
    else:
        data[idx][2] = random.randint(0,1)

edge_labels = {}
G = nx.Graph()
# G.add_nodes_from(users)
G.add_edges_from(edges)
a = []
for idx,(i,x,attr) in enumerate(G.edges(data=True)):
    a.append((i,x))
    if data[idx][2] == 1:
        G.add_edge(i,x, sign = '+')
    else:
        G.add_edge(i,x,sign = '-')

    edge_labels[i,x] = attr['sign']

def set_network():
    edge_labels = {}
    G = nx.Graph()
    G.add_nodes_from(users)
    G.add_edges_from(edges)
    a = []
    for idx,(i,x,attr) in enumerate(G.edges(data=True)):
        a.append((i,x))
        if data[idx][2] == 1:
            G.add_edge(i,x, sign = '+')
        else:
            G.add_edge(i,x,sign = '-')

        # print(i, x, attr)
        edge_labels[i,x] = attr['sign']

    return G, edge_labels

pos = nx.spring_layout(G)
nx.draw_networkx(G,pos)
nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_color='red',font_weight='bold')
plt.show()