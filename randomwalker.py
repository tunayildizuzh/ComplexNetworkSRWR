import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class RandomWalker:

    """" Signed Random Walker with Reset agent class.

        Parameters:

        position = int
        Current position of the Signed Random Walker

        sign = string
        + or - depending on the node the agent is on.

        c: float
        Determines the rate of jump to the rate of reset
    """

    def __init__(self,position, sign,c):

        self.position = position
        self.sign = sign
        self.c = c



class SingleSimulation:

    """
    Walking of a single signed random walker.

    Parameters:

        random_walker: RandomWalker
            An instance of a Signed Random Walker with Reset

        G: nx.Graph
            Networkx Graph where the walkers jumps through

        lambda_: float
            rate of the walk. Chosen from an exponential distribution

        t_start: float
            Starting of the simulation

        t_end: float
            End of the simulation
    """

    def __init__(self, random_walker, G, t_start = 0, t_end = 10):

        self.random_walker = random_walker
        self.G = G
        self.t_start = t_start
        self.t_end = t_end
        self.path = []



    def get_next_event(self):

        next_pos = random.choice(list(self.G.neighbors(self.random_walker.position)))

        if next_pos == self.random_walker.position:
            self.get_next_event()

        return next_pos

    def signed_adjacency_matrix(self):

        A = nx.to_numpy_matrix(self.G)

        sign = nx.get_edge_attributes(self.G,'sign')

        for key in sign:
            if sign[key] == '-':
                A[key] = -1
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
        # print('APLUS A NEG')
        # print(A_plus)
        # print(A_negative)
        delta = 0
        q = np.zeros(self.G.number_of_nodes())
        q[seed_node] = seed_node
        # print('Q')
        # print(q)
        r_plus = q
        # print(r_plus)
        r_negative = np.zeros(self.G.number_of_nodes())
        # print(r_negative)
        r_prime = np.vstack((r_plus,r_negative))
        # r_prime = np.concatenate((r_plus, r_negative))
        # print('RPRIME')
        # print(r_prime)
        while np.all(delta) < epsilon:
            r_plus = (1-c) * ((np.matmul(A_plus,r_plus) + (beta * np.matmul(A_negative,r_negative)) + ((1-gamma) * np.matmul(A_plus,r_negative))) + c*q)
            # print("R PLUSS")
            # print(r_plus)
            r_negative = (1-c) * (np.matmul(A_negative,r_plus) + (gamma * np.matmul(A_plus,r_negative)) + ((1-beta) * np.matmul(A_negative,r_negative)))
            # print('R NEGG')
            # print(r_negative)
            r = np.vstack((r_plus,r_negative))
            # print(r)
            delta = abs(r - r_prime)
            # print('DELTA')
            # print(delta)
            r_prime = r
        print('Done!')
        print(r_plus)
        print(r_negative)
        print('rp - rn')
        print(r_plus + r_negative)
        return r_plus, r_negative

    def run(self,target_node):
        visited = []
        pos_init = self.random_walker.position
        visited.append(pos_init)
        idx = 0
        # while self.t_start <= self.t_end:

        while target_node != self.random_walker.position:

            reset_prop = random.uniform(0,1)
            if reset_prop < self.random_walker.c: # Reset Case.
                self.random_walker.position = pos_init
                self.random_walker.sign = '+'
                print(f'Reset. Current position: {pos_init}')
                visited = [visited[0]]

            else: # Jump Case.

                next_position = self.get_next_event()

                if next_position not in visited:
                    visited.append(next_position)
                    # (self,seed_node, current_position, next_position, edge_sign, walker_sign, degree):
                    self.calculate_r(0,self.random_walker.position,next_position,self.G.edges[self.random_walker.position, next_position]['sign'],self.random_walker.sign,self.G.degree(self.random_walker.position))

                    if self.G.edges[self.random_walker.position, next_position]['sign'] == '-':
                        if self.random_walker.sign == '+':
                            self.random_walker.sign = '-'
                        elif self.random_walker.sign == '-':
                            self.random_walker.sign = '+'
                    elif self.G.edges[self.random_walker.position, next_position]['sign'] == '+':
                        if self.random_walker.sign == '+':
                            self.random_walker.sign = '+'
                        elif self.random_walker.sign == '-':
                            self.random_walker.sign = '-'

                    # self.calculate_r(0,self.random_walker.position,next_position,self.G.edges[self.random_walker.position, next_position]['sign'],self.random_walker.sign,self.G.degree(self.random_walker.position))

                    print(f"From: {self.random_walker.position}, To: {next_position}, Edge Sign: {self.G.edges[int(self.random_walker.position), int(next_position)]['sign']} ")
                    self.random_walker.position = next_position
                    self.t_start += 1
                else:
                    continue
                print(f'SRWR Sign: {self.random_walker.sign}')


