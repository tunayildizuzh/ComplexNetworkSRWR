import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def set_G():
    data = np.genfromtxt("/Users/tunayildiz/Desktop/UZH/ComplexNetworkSRWR/dataset/BTCAlphaNet.csv", delimiter=",", dtype=float)[0:50]
    np.random.shuffle(data)


    users = []
    edges = []
    seed_node = data[0,0]
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

    return G, edge_labels, seed_node, users, edges


# pos = nx.spring_layout(set_G()[0])
# nx.draw_networkx(set_G()[0],pos)
# nx.draw_networkx_edge_labels(set_G()[0],pos,edge_labels=set_G()[1],font_color='red',font_weight='bold')
# plt.show()