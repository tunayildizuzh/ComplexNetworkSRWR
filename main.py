import networkx as nx
import matplotlib.pyplot as plt
import random
from randomwalker import SingleSimulation
from randomwalker import RandomWalker
import scipy


def set_watts_strogatz(n, k, p): # Create a Watts Strogatz Network and assign signs to its edges.
    # G.edges[0, 1]['sign']
    edge_labels = {}
    node_labels = {}
    G = nx.watts_strogatz_graph(n, k, p)

    for i,x,attr in G.edges(data=True):
        val = random.uniform(0,1)
        if val <= 0.5:
            G.add_edge(i,x, sign = '+')

        else:
            G.add_edge(i,x,sign = '-')

        edge_labels[i,x] = attr['sign']

    nx.set_node_attributes(G,1,'r')

    return G, edge_labels


graph = set_watts_strogatz(7,5,0.05)
single_simulation = SingleSimulation(RandomWalker(0,'+',0.1),graph[0],0,5)
pos = nx.spring_layout(single_simulation.G)
nx.draw_networkx(single_simulation.G,pos,with_labels=True)
nx.draw_networkx_edge_labels(single_simulation.G,pos,edge_labels=graph[1],font_color='red',font_weight='bold')
single_simulation.run(4)

plt.show()