#!/usr/bin/env python
__author__ = "Guillaume Bellec"

'''
    File name: gradient_descent_finite_difference.py
    Author: Guillaume Bellec
    Date created: 8/29/2016
    Date last modified: 18/09/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    This file contains the implementation of a generic solver based on gradient descent with estimation of the gradient
    through finite differences.
    In the course Autonomous Learning Systems, students should in this file:
    - Fill the TODOs to finish to implementation
'''

import numpy as np
import numpy.random as rd
from time import time


class GDFDSolver:
    def __init__(self, learning_rate, exploration_rate, n_random_step, n_iteration, stop_criterion=-np.Inf):
        '''
        Generic solver based on gradient descent and estimation of the gradient with finite differences.

        :param learning_rate: learning rate as defined in standard gradient descent
        :param exploration_rate: size of the neighbourhood used to estimate the gradient
        :param n_random_step: number of neighbours to estimate the gradient
        :param n_iteration: number of iterations
        '''
        self.n_iteration = n_iteration
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.n_random_steps = n_random_step
        self.stop_criterion = stop_criterion

    def solve(self, x0, fun, bound):
        '''
        Method to actually minimize the cost function fun starting at the point x0.

        :param x0: Starting parameter set
        :param fun: Function so minimize
        :param bound: Bound of the admissible space
        :return:
        '''

        # init
        x = x0.copy()
        f = np.Inf

        t0 = time()
        function_call = 1
        f_list = []
        x_list = []

        for k in range(self.n_iteration):
            # New step, and estimate the gradient

            # Compute the function at the current point
            f = fun(x)
            function_call += 1

            #------------------------
            # TODO:
            #   1) Compute self.n_random_steps small random perturbation dx of the parameters
            #   Each should be taken from a gaussian distribution around zero.
            #   The standard deviation of this gaussian should be proportional to the exploration step size.
            #
            #   2) Compute the function at each x + dx,
            #   (Tip: do not forget to count each call to fun into function_call)
            #
            #   3) Estimate the best possible approximation of the gradient (see slides of the practical)
            #
            #   4) Make a gradient descent step of size learning_rate in the direction of the gradient
            #   (Tip: clip the new solution within the admissible bounds
            #

            # dx = np.zeros((self.n_random_steps, x0.size)) # Replace me but the size should match
            dx = rd.normal(loc=0.0, scale=self.exploration_rate, size=(self.n_random_steps, x0.size))

            f_ = np.zeros(self.n_random_steps) # Replace me but the size should match
            for i in range(0, self.n_random_steps):
                f_[i] = fun(x + dx[i])
                function_call += 1

            A = np.zeros(dx.shape)
            A[:] = dx[:] - x
            df = np.zeros(f_.shape)
            df[:] = f_[:] - f
            g = np.zeros(x.shape) # Replace me but the size should match
            g = np.dot(np.linalg.pinv(A), df)

            x = x - self.learning_rate * g
            x[0] = max(min(x[0], bound[1]), bound[0])
            x[1] = max(min(x[1], bound[1]), bound[0])

            #-----------------------

            f_list.append(f)
            x_list.append(x)

            if f_list[-1] <= self.stop_criterion:
                break

        t = time() - t0
        print('Grad Descent Finite Diff: \n\t f={:.3g} \t iteration: {} \t time: {:.3g} s \t function call: {}'.format(
            f_list[-1], k, t, function_call))
        result = {
            'x_list': x_list,
            'f_list': f_list,
            'function_call': function_call,
            'time': t,
            'iteration': k
        }
        return result
