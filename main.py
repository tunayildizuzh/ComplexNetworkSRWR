import networkx as nx
import matplotlib.pyplot as plt
import random
from randomwalker import Simulation
import numpy as np
import scipy
from readdata import set_G
import pandas as pd
import seaborn as sns
from collections import OrderedDict
from readdata import data_settings


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



def comparison_dict(seed_node,rd_val,graph_nodes): # SRWR Scores of the seed node to the other nodes.
    comparison = {}

    for idx, val in enumerate(graph_nodes):
        comparison[seed_node,val] = rd_val[idx]

    for a,b in comparison:
        if comparison[a,b] > 0:
            comparison[a,b] = '+'
        else:
            comparison[a,b] = '-'


    return comparison


def accuracy(seed_node,rd_val,graph_nodes,edge_attributes): # SRWR sign prediction comparison to real data signs.
    count = 0
    true = 0

    comparison = comparison_dict(seed_node,rd_val,graph_nodes)
    for a,b in edge_attributes:
            if a == seed_node:
                count+=1
                if edge_attributes[a,b] == comparison[a,b]:
                    true+=1
    accuracy_percentage = (true/count) * 100

    return accuracy_percentage



"""
Part 1: Working with a synthetic network to check if SRWR works.
"""
#
# graph = set_watts_strogatz(10,4,0.05)
# simulation = Simulation(graph[0],seed_node=1,c=0.15,beta=0.1,gamma=0.1,epsilon=0.1)


# print(f"Initial State: {nx.get_edge_attributes(simulation.G,'sign')}")
# print(simulation.nodes)
# simulation.remove_edge(simulation.seed_node,0.5)

# simulation.normalize()
# iterate = simulation.iterate(simulation.normalize()[0],simulation.normalize()[1],simulation.seed_node)
# rd = iterate[1] - iterate[0]

# edge_attr = nx.get_edge_attributes(simulation.G,'sign')
# print(f'Accuracy Percentage: {accuracy(simulation.seed_node,rd,simulation.nodes,edge_attr)}%')
# print(edge_attr)

# pos = nx.spring_layout(simulation.G)
# nx.draw_networkx(simulation.G,pos,with_labels=True)
# nx.draw_networkx_edge_labels(simulation.G,pos,edge_labels=edge_attr,font_color='red',font_weight='bold')
# plt.show()
#



"""
Part 2: Use a real network for SRWR application.
"""

graph_network = set_G()
def run(beta, gamma, epsilon,step=0): # Runs the Simulation.
    if step == 1:
        print(f'Seed Node: {graph_network[2]}')
    simulate = Simulation(graph_network[0],seed_node=graph_network[2],c=0.15,beta=beta,gamma=gamma,epsilon=epsilon)
    edge_attributes = nx.get_edge_attributes(simulate.G, 'sign')
    if step == 1:
        simulate.remove_edge(simulate.seed_node,0.1)
    simulate.normalize()
    iterate = simulate.iterate(simulate.normalize()[0],simulate.normalize()[1],graph_network[2])
    rd = iterate[1] - iterate[0]
    print(f'Accuracy Percentage: {accuracy(graph_network[2],rd,simulate.nodes,edge_attributes)}%')
    return accuracy(graph_network[2],rd,simulate.nodes,edge_attributes)


parameter_values = np.arange(1,11,1)
accuracies = np.zeros((10,10))
step_iter = 1
for index,value in enumerate(parameter_values): # X = beta Y = Gamma. Looks for the best beta and gamma values
    for index2,value2 in enumerate(parameter_values):
        print(f'beta,gamma: {value,value2}')
        accuracies[index,index2] = run(value,value2,0.1,0)
        step_iter = 0

df = pd.DataFrame(accuracies)
ax = sns.heatmap(df, annot = True)
ax.set(xlabel='Beta',ylabel='Gamma')
plt.savefig('accuracies_parameter.png')

param = np.where(accuracies == np.amax(accuracies))
print(f'PARAM: {param}')

run(param[0][0],param[1][0],0.1,1) # Runs the simulation again with the most optimal beta and gamma values to get highest accuracy.


# data_settings() # For plots
