import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from readdata import set_G
from random import sample
from collections import OrderedDict
users = set_G()[3]
edges = set_G()[4]


class Simulation:
    '''
    SRWR Simulation.

    G: Networkx Graph.

    seed_node: int
        The node that the walk starts. Picked by degree centrality hierarchy.

    c: float
        The restart probability of the walker.

    beta: float
        Balance attenuation factor. Parameter of the uncertainty of 'the enemy of my enemy is my friend'.

    gamma: float
        Balance attenuation factor. Parameter of the uncertainty of 'the friend of my enemy is my enemy'.

    epsilon: float
        Walk iteration continues until convergence is smaller than epsilon.
    '''


    def __init__(self, G, seed_node,c,beta,gamma,epsilon):
        self.seed_node = seed_node
        self.G = G
        self.nodes = list(self.G.nodes())
        self.c = c
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def signed_adjacency_matrix(self):
        global nodes
        nodes = list(self.G.nodes())
        A = nx.to_numpy_matrix(self.G)
        sign = nx.get_edge_attributes(self.G,'sign')
        for key in sign:
            if sign[key] == '-':
                A[int(nodes.index(key[0])),int(nodes.index(key[1]))] = -1
                A[int(nodes.index(key[1])), int(nodes.index(key[0]))] = -1
            if sign[key] != '+' and sign[key] != '-':
                A[int(nodes.index(key[0])), int(nodes.index(key[1]))] = 0
                A[int(nodes.index(key[1])), int(nodes.index(key[0]))] = 0
        return A

    def normalize(self):
        degrees = [val for (node, val) in self.G.degree()]
        D = np.diag(degrees)
        semi_row_normalized_matrix = np.matmul(np.linalg.inv(D), self.signed_adjacency_matrix())
        A_plus = np.zeros(semi_row_normalized_matrix.shape)
        A_negative = np.zeros(semi_row_normalized_matrix.shape)

        for x in range(semi_row_normalized_matrix.shape[0]):
            for y in range(semi_row_normalized_matrix.shape[1]):
                if semi_row_normalized_matrix[x,y] > 0:
                    A_plus[x,y] = semi_row_normalized_matrix[x,y]
                if semi_row_normalized_matrix[x,y] < 0:
                    A_negative[x,y] = semi_row_normalized_matrix[x,y]

        return A_plus, A_negative

    def iterate(self,A_plus, A_negative, seed_node):

        nodes = list(self.G.nodes())
        q = np.zeros(self.G.number_of_nodes())
        q[nodes.index(seed_node)] = 1
        r_plus = q
        r_negative = np.zeros(self.G.number_of_nodes())
        r_prime = np.vstack((r_plus,r_negative))


        for i in range(70):
        # while np.all(delta<self.epsilon) == False:

            # r_plus = (1-c) * ((np.matmul(A_plus,r_plus)) + (np.matmul(A_negative,r_negative))) + c*q
            # r_negative = (1-c) * ((np.matmul(A_negative,r_plus)) + (np.matmul(A_plus,r_negative)))
            r_plus = ((1-self.c) * (np.matmul(A_plus,r_plus) + (self.beta * np.matmul(A_negative,r_negative)) + ((10-self.gamma) * np.matmul(A_plus,r_negative)))) + self.c*q
            r_negative = (1-self.c) * ((np.matmul(A_negative,r_plus) + (self.gamma * np.matmul(A_plus,r_negative)) + ((10-self.beta) * np.matmul(A_negative,r_negative))))
            r = np.vstack((r_plus,r_negative))
            delta = abs(r - r_prime)
            # print(f'DELTA: {delta}')
            # print(f'delta:{delta}')
            r_prime = r


        return r_plus, r_negative

    def remove_edge(self,node=0, percent = 0.2): # Removes the edges of a node by percentage of their total edge count.
        degree = sorted(self.G.degree, key=lambda asd: asd[1], reverse=True)
        print(f'remove edge degree: {degree[0]}')

        deleted_edge_count = int(degree[0][1] * percent * 0.7)
        print(f'DELETED {deleted_edge_count}')
        # neighs = list(nx.neighbors(self.G, node))
        neighs = list(self.G.neighbors(node))
        random_neighbor_list = sample(neighs,deleted_edge_count)
        for random_neighbor in random_neighbor_list:
            if int(node) < int(random_neighbor):
                if (node,random_neighbor) in nx.get_edge_attributes(self.G,'sign'):
                    if nx.get_edge_attributes(self.G,'sign')[int(node),int(random_neighbor)] != '?':
                        self.G.add_edge(int(node),int(random_neighbor),sign = '?')
                    else:
                        continue
                else:
                    continue

            if int(node) > int(random_neighbor):
                if (node, random_neighbor) in nx.get_edge_attributes(self.G, 'sign'):
                    if nx.get_edge_attributes(self.G,'sign')[int(random_neighbor),int(node)] != '?':
                        self.G.add_edge(int(random_neighbor),int(node),sign = '?')
                    else:
                        continue
                else:
                    continue








