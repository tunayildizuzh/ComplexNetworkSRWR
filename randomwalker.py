import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from readdata import set_G
from collections import OrderedDict
users = set_G()[3]
edges = set_G()[4]


class Simulation:

    def __init__(self, G, seed_node):
        self.seed_node = seed_node
        self.G = G
        self.nodes = list(self.G.nodes())

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

    def iterate(self,A_plus, A_negative, seed_node, c, beta, gamma, epsilon):

        nodes = list(self.G.nodes())
        delta = 0
        q = np.zeros(self.G.number_of_nodes())
        q[nodes.index(seed_node)] = 1
        print(f'q:{q}')
        r_plus = q
        r_negative = np.zeros(self.G.number_of_nodes())
        r_prime = np.vstack((r_plus,r_negative))
        # print(f'r_prime:{r_prime}')
        # epsilon_np = np.zeros((r_prime.shape))
        # epsilon_np[:] = epsilon

        for i in range(50):
        # while np.all(delta<epsilon):


            # r_plus = (1-c) * ((np.matmul(A_plus,r_plus)) + (np.matmul(A_negative,r_negative))) + c*q
            # r_negative = (1-c) * ((np.matmul(A_negative,r_plus)) + (np.matmul(A_plus,r_negative)))
            r_plus = ((1-c) * (np.matmul(A_plus,r_plus) + (beta * np.matmul(A_negative,r_negative)) + ((1-gamma) * np.matmul(A_plus,r_negative)))) + c*q
            r_negative = (1-c) * ((np.matmul(A_negative,r_plus) + (gamma * np.matmul(A_plus,r_negative)) + ((1-beta) * np.matmul(A_negative,r_negative))))
            r = np.vstack((r_plus,r_negative))
            delta = abs(r - r_prime)
            # print(f'delta:{delta}')
            r_prime = r

        # print(f'r_plus:{r_plus}')
        # print(f'r_negative:{r_negative}')
        print(f"rp-rn: {r_plus - r_negative}")
        return r_plus, r_negative

    def remove_edge(self,node=0):

        neighs = list(nx.neighbors(self.G, node))
        random_neighbor = random.choice(neighs)

        if int(node) < int(random_neighbor):
            if nx.get_edge_attributes(self.G,'sign')[node,random_neighbor] != '?':
                self.G.add_edge(node,random_neighbor,sign = '?')
            else:
                self.remove_edge()
        if int(node) > int(random_neighbor):
            if nx.get_edge_attributes(self.G,'sign')[random_neighbor,node] != '?':
                self.G.add_edge(random_neighbor,node,sign = '?')
            else:
                self.remove_edge()








