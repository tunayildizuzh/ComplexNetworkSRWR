import networkx as nx
import matplotlib.pyplot as plt
import random
from randomwalker import Simulation
import scipy
from readdata import set_G
from collections import OrderedDict


def set_watts_strogatz(n, k, p): # Create a Watts Strogatz Network and assign signs to its edges.
    # G.edges[0, 1]['sign']
    edge_labels = {}
    edge_new = {}
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
    # print('Edge Attributes:')
    # print(nx.get_edge_attributes(G,'sign'))
    edge_attr = nx.get_edge_attributes(G,'sign')
    # print(f'edgelabels: {edge_labels}')
    return G, edge_attr

"""
Part 1: Working with a synthetic network to check if SRWR works.
"""
#
# graph = set_watts_strogatz(4,4,0.05)
# simulation = Simulation(graph[0],seed_node=1)
#
#
# print(f"Initial State: {nx.get_edge_attributes(simulation.G,'sign')}")
# print(simulation.nodes)
# simulation.remove_edge(1)
#
# simulation.normalize()
# simulation.iterate(simulation.normalize()[0],simulation.normalize()[1],simulation.seed_node,0.15,0.1,0.1,0.005)
# simulation.remove_edge(simulation.seed_node)
# edge_attr = nx.get_edge_attributes(simulation.G,'sign')
# print(edge_attr)
#
#
# pos = nx.spring_layout(simulation.G)
# nx.draw_networkx(simulation.G,pos,with_labels=True)
# nx.draw_networkx_edge_labels(simulation.G,pos,edge_labels=edge_attr,font_color='red',font_weight='bold')
# plt.show()




"""
Part 2: Use a real network for SRWR application.
"""
graph2 = set_G()

print(f'Seed Node: {graph2[2]}')
sim2 = Simulation(graph2[0],seed_node=graph2[2])
edge_attributes = nx.get_edge_attributes(sim2.G,'sign')
print(f"Initial State: {edge_attributes}")
print(sim2.nodes)
# print(f'edges:{graph2[4]}')

sim2.normalize()
sim2.iterate(sim2.normalize()[0],sim2.normalize()[1],graph2[2], 0.15,0.3,0.5,0.1)
# sim2.remove_edge(node=graph2[2])
# sim2.remove_edge(node=graph2[2])
edge_attr = nx.get_edge_attributes(sim2.G,'sign')
rd = sim2.iterate(sim2.normalize()[0],sim2.normalize()[1],graph2[2], 0.15,0.3,0.5,0.1)[0] - sim2.iterate(sim2.normalize()[0],sim2.normalize()[1],graph2[2], 0.15,0.3,0.5,0.1)[1]
zipped_values = list(zip(sim2.nodes,rd))
print(f'ZIPPED: {zipped_values}')
edge_attr_sim2 = nx.get_edge_attributes(sim2.G,'sign')
print(f"Edge Attributes: {edge_attr_sim2}")


pos = nx.spring_layout(sim2.G)
nx.draw_networkx(sim2.G,pos,with_labels=True)
nx.draw_networkx_edge_labels(sim2.G,pos,edge_labels=edge_attr,font_color='red',font_weight='bold')
plt.show()





