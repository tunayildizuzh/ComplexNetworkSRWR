import random
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



    def get_next_event(self):

        next_pos = random.choice(list(self.G.neighbors(self.random_walker.position)))
        if next_pos == self.random_walker.position:
            self.get_next_event()

        return next_pos

    def run(self):
        pos_init = self.random_walker.position

        while self.t_start <= self.t_end:

            reset_prop = random.uniform(0,1)
            if reset_prop < self.random_walker.c: # Reset Case.
                self.random_walker.position = pos_init
                self.random_walker.sign = '+'
                print(f'Reset. Current position: {pos_init}')

            else: # Jump Case.

                next_position = self.get_next_event()
                if self.G.edges[self.random_walker.position, next_position]['sign'] == '-':
                    if self.random_walker.sign == '+':
                        self.random_walker.sign = '-'
                    elif self.random_walker.sign == '-':
                        self.random_walker.sign = '+'

                print(f"From: {self.random_walker.position}, To: {next_position}, Edge Sign: {self.G.edges[self.random_walker.position, next_position]['sign']} ")
                self.random_walker.position = next_position


            self.t_start += 1
            print(f'Current Sign: {self.random_walker.sign}')






