#!/usr/bin/env python
__author__ = "Guillaume Bellec"

'''
    File name: simulated_annealing.py
    Author: Guillaume Bellec
    Date created: 8/29/2016
    Date last modified: 18/09/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    This file contains the definition of a generic and simple implementation of Simulated Annealing.
    In the course Autonomous Learning Systems, students should in this file:
    - Fill the TODOs to finish to implementation
'''

import numpy as np
import numpy.random as rd
from time import time


class AnnealSolver:
    def __init__(self, noisy_step, temp_decay, n_iteration, stop_criterion=- np.Inf):
        '''
        Class for a generic simulate annealing solver.

        In the pseudo code the algorithm does:

        For n iterations do:
            - Take a step of size noisy step in a random direction
            - If it reduces the cost, keep the solution
            - Otherwise keep with probability exp(- (f_new - f) / T)

        :param noisy_step: Size of the random step
        :param temp_decay: A function of the form f(t) = temperature at time t
        :param n_iteration: number of iteration to perform
        '''

        self.noisy_step = noisy_step
        self.temp_decay = temp_decay
        self.n_iteration = n_iteration
        self.stop_criterion = stop_criterion

    def solve(self, x0, fun, bound):
        '''
        Actual function that performs the optimization loop.

        :param x0: Inital solution
        :param fun: Function to minimize
        :param bound: Bound [x_min,x_max] of the solution space assuming that all coordinates must stay within [x_min,x_max]
        :return:
        '''

        x = x0  # Initialize of the running solution
        f = np.Inf  # Initialize of the cost function
        T = 1. # Initialize temperature

        t0 = time()  # Initialize the clock
        n_function_call = 0  # Initialize the count of function calls

        f_list = []  # Initialize the list of fitness
        x_list = []  # Initialize the list of fitness

        for k in range(self.n_iteration):

            #------------------------
            # TODO:
            #   Make step in a random direction and update the temperature
            #
            # Tips:
            #   - Make sure to stay in the solution space by clipping x into the bounds: bound = [x_min,x_max]
            #
            step = self.noisy_step
            x_new = np.zeros(x.shape)
            random_step = np.array([rd.uniform(-step, step) for i in range(x.size)])
            x_new[:] = x[:] + random_step[:]
            bound_0 = np.zeros(x.shape)
            bound_0.fill(bound[0])
            bound_1 = np.zeros(x.shape)
            bound_1.fill(bound[1])
            x_new = np.maximum(np.minimum(x_new, bound_1), bound_0)
            T *= self.temp_decay
            #-----------------------

            f_new = fun(x_new)  # Compute the function at the new solution
            n_function_call += 1 # Make sure to count the function call

            # ------------------------
            # TODO:
            #  Accept of reject the new solution depending on the temperate and the possible improvment of the solution.
            #  (WARNING: Make sure to have the sign right, we are doing minimization not maximization as in the lecture)
            #
            if f_new < f:  ## Replace me with acceptance condition
                x = x_new   ## Replace me
                f = f_new
            else:
                if rd.random() < np.exp(-(f_new - f)/T):
                    x = x_new   ## Replace me
                    f = f_new
            # -----------------------

            # Keep all intermediate solutions in memory to debug and report the results nicely
            f_list.append(f)
            x_list.append(x)

            if f_list[-1] <= self.stop_criterion:
                break

        # Get the time of computation for comparison with other algorithms
        t = time() - t0
        print('Simulated Annealing: \n\t f={:.3g} \t iteration: {} \t time: {:.3g} s \t function call: {}'\
              .format(f, k, t, n_function_call))

        # Put all information useful for comparison with other algorithms in a dictionary
        result = {
            'x_list': x_list,
            'f_list': f_list,
            'n_function_call': n_function_call,
            'time': t,
            'iteration': k
        }
        return result
