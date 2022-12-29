import networkx as nx
import matplotlib.pyplot as plt
import random
from randomwalker import SingleSimulation
from randomwalker import RandomWalker


def set_watts_strogatz(n, k, p): # Create a Watts Strogatz Network and assign signs to its edges.
    # G.edges[0, 1]['sign']
    edge_labels = {}
    G = nx.watts_strogatz_graph(n, k, p)

    for i,x,attr in G.edges(data=True):
        val = random.uniform(0,1)
        if val <= 0.5:
            G.add_edge(i,x, sign = '+')
        else:
            G.add_edge(i,x,sign = '-')

        edge_labels[i,x] = attr['sign']

    return G, edge_labels

def plot_G():
    a = set_watts_strogatz(10,6,0.6)
    G = a[0]
    pos = nx.spring_layout(G)
    nx.draw_networkx(G,pos,with_labels=True)
    nx.draw_networkx_edge_labels(G,pos, edge_labels = a[1], font_color = 'red', font_weight='bold')
    plt.show()

single_simulation = SingleSimulation(RandomWalker(0,'+',0.15),set_watts_strogatz(20,8,0.4)[0],0,10)
single_simulation.run()