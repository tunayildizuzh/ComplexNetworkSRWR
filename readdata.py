import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
import collections
import random

def set_G():
    data = np.genfromtxt("/Users/tunayildiz/Desktop/UZH/ComplexNetworkSRWR/dataset/BTCAlphaNet.csv", delimiter=",", dtype=float)
    np.random.shuffle(data)
    # print(f'DATA0: {data[0]}')

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

    degree = sorted(G.degree, key=lambda asd: asd[1], reverse=True)
    # print(degree)
    seed_node = degree[0][0]
    # print(seed_node)


    return G, edge_labels, seed_node, users, edges


def data_settings():
    degrees = []
    degree_count = []
    G = nx.Graph(set_G()[0])
    # print(list(G.degree()))

    '''Degree Distribution'''
    for node, deg in  list(G.degree()):
        degrees.append(deg)
    print(degrees)

    for i in range(94):

        degree_count.append(degrees.count(i))

    print(degree_count)

    plt.plot(np.arange(1,95,1), degree_count)
    plt.xlabel('Count')
    plt.ylabel('Degree')
    plt.title('Degree Distribution')
    plt.savefig('degree_distribution.png')



    '''CCDF Plot '''
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cs = np.cumsum(cnt)
    plt.loglog(deg, cs, 'bo')
    plt.title("Cumulative Distribution plot")
    plt.ylabel("P(K>=k)")
    plt.xlabel("k")
    plt.savefig('CCDF.png')
    plt.show()







