#!/usr/bin/env python
__author__ = "Guillaume Bellec"

'''
    File name: gradient_descent_finite_difference.py
    Author: Guillaume Bellec
    Date created: 8/29/2016
    Date last modified: 18/09/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    This file contains a generic implementation of an evolutionnary algorithm solver
    In the course Autonomous Learning Systems, students should in this file:
    - Fill the TODOs to finish to implementation
'''

import numpy as np
import numpy.random as rd
from time import time
from itertools import product


class GeneticSolver:
    def __init__(self, selection_temperature, mutation_rate, crossover_rate, n_iteration, stop_criterion=-np.Inf):
        '''
        Generic solver to implement a genetic Algorithm.

        :param selection_temperature:
        :param mutation_rate:
        :param crossover_rate:
        :param n_iteration:
        '''
        self.n_iteration = n_iteration

        self.selection_temperature = selection_temperature
        self.mutation_rate = mutation_rate
        self.reproduction_rate = crossover_rate
        self.stop_criterion = stop_criterion

    def mutation(self, pop, bound):
        '''
        Mutation function. It takes the current population of admissible parameters and the bound of the domain.
        It should return a population a muted parameter sets.

        :param pop:
        :param bound:
        :return:
        '''

        # ------------------------
        # TODO:
        #   Implement the uniform mutation as explained in the practical slides.
        #   The probability to have one or more coefficient that mutes is self.mutation_rate for each parameter in pop
        #   (Tip: try to use numpy built-in function and avoid for loops to save computation time)
        #

        if rd.uniform() < self.mutation_rate:
            pop[0] = rd.uniform(bound[0], bound[1])
        if rd.uniform() < self.mutation_rate:
            pop[1] = rd.uniform(bound[0], bound[1])

        # -----------------------

        return pop

    def crossover(self, pop):
        '''
        Implement the crossover within the population pop.
        It should return the population that received the crossover.

        :param pop:
        :return:
        '''
        shp = pop.shape

        # ------------------------
        # TODO:
        #   Implement the arithmetic crossover as explained in the practical slides.
        #   Crossover between each pair of individual happens with probability self.crossover_rate
        #   (Tip: for instance, one can first copy the population and create a third that is the crossover of the first
        #   two)
        #

        src1 = pop.copy()
        src2 = pop.copy()
        for i, i_1 in enumerate(src1):
            for j, i_2 in enumerate(src2):
                if rd.uniform() < self.reproduction_rate:
                    r = rd.uniform()

                    x_1 = r * i_1[0] + (1-r) * i_2[0]
                    y_1 = r * i_1[1] + (1-r) * i_2[1]
                    pop[i] = [x_1, y_1]

                    x_2 = (1-r) * i_1[0] + r * i_2[0]
                    y_2 = (1-r) * i_1[1] + r * i_2[1]
                    pop[j] = [x_2, y_2]

        # -----------------------

        return pop

    def selection(self, pop, f_pop):
        '''
        Selection function.
        It take the population and the cost function of each parameters.
        It returns a population of same size where the fittest elements appear with higher probability.

        :param pop:
        :param f_pop:
        :return:
        '''

        # ------------------------
        # TODO:
        #   Implement the softmax selection.
        #   Each element fot the output population is taken from the initial population with probability that if the
        #   softmax of the ratio: f(x) divided by 'self.selection_temperature'.
        #
        #   WARNING:
        #       1) Make sure you get the sign right. Note that here we do a minimization of the cost function whereas
        #       the slides described a maximization of the fitness,
        #       2) Make sure to also re-order the cost-functions in f_pop accordingly.
        #

        new_pop = []
        new_f_pop = []
        exp_sum = np.sum([np.exp(f/self.selection_temperature) for f in f_pop])
        for i, ind in enumerate(pop):
            p = np.exp(f_pop[i]/self.selection_temperature) / exp_sum
            # FIXME: should be <, but probabilities too small?
            # FIXME: probably in combination with sign (see WARNING)
            if rd.uniform() > p:
                new_pop.append(ind)
                new_f_pop.append(f_pop[i])

        if len(new_pop) < pop.size:
            new_pop = np.array(new_pop)
            missing = pop.shape[0] - new_pop.shape[0]
            duplicate_list = [1 for i in range(new_pop.shape[0] - 1)]
            duplicate_list.append(missing+1)
            new_pop = np.repeat(new_pop, duplicate_list, axis=0)

        new_f_pop = np.array(new_f_pop)

        # -----------------------

        return new_pop, new_f_pop

    def solve(self, pop, fun, bound):
        '''
        Method to run the successive iteration of the genetic algorithm.

        :param pop: initial population of code strings (i.e population of parameter candidates)
        :param fun: function to minimize
        :param bound: admissible bounds
        :return:
        '''
        pop = np.array(pop)

        # init
        n_pop = pop.shape[0]  # number of parameter in each population
        function_call = 0 # count the number of call to the cost-function
        t0 = time() # Init the clock

        x_list = [] # keep track of the optimal parameters
        f_list = [] # keep track of the cost of the optimal parameters

        pop_list = [] # Keep track of the list of population parameters
        f_pop_list = [] # Keep track of their costs

        f_pop = [np.Inf for x in pop] # Initialize the cost of the population to an arbitrary value

        for k in range(self.n_iteration):
            # Mutation and Crossover
            pop = self.mutation(pop, bound)
            pop = self.crossover(pop)

            # Compute cost_function
            f_pop = np.array([fun(pop[k, :]) for k in range(n_pop)])
            function_call += n_pop

            # Save the population before selection
            pop_list.append(pop.copy())
            f_pop_list.append(f_pop.copy())

            # Selection
            pop, f_pop = self.selection(pop, f_pop)

            # Save one of the best candidates
            x_list.append(pop[0, :].copy())
            f_list.append(f_pop[0])

            if f_list[-1] <= self.stop_criterion:
                break

        t = time() - t0
        print('Genetic algorithm: \n\t f={:.3g} \t iteration: {} \t time: {:.3g} s \t function call: {}'.format(
            f_list[-1], k, t, function_call))
        result = {
            'x_list': x_list,
            'f_list': f_list,
            'pop_list': np.array(pop_list),
            'f_pop_list': np.array(f_pop_list),
            'function_call': function_call,
            'time': t,
            'iteration': k
        }
        return result
