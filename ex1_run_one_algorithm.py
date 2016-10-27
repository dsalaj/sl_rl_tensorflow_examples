#!/usr/bin/env python
__author__ = "Guillaume Bellec"

'''
    File name: ex1_run_one_algorithm.py
    Date created: 8/29/2016
    Date last modified: 10/14/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    Usage of optimization algorithms to solve standard non-differential or non-convexe functions

    Follow the instructions of the hand-out.
    Only solver parameters need to be modified in this file.

    To run this file students should:
    - Complete the TODOs in 'simulated_annealing.py'
    - Complete the TODOs in in 'gradient_descent_finite_differences.py'
    - Complete the TODOs in in 'genetic'.py
    - Run this file to test you results
    Follow the instruction on the hand-out.

'''

import numpy as np
import numpy.random as rd
from ex1_cost_functions import get_cost_function, plot_cost
import matplotlib.pyplot as plt
from simulated_annealing import AnnealSolver
from gradient_descent_finite_difference import GDFDSolver
from genetic import GeneticSolver
from mpl_toolkits.mplot3d import Axes3D

# Choose the function to minimize by comment the others
# cost_name = 'rastrigin'
cost_name = 'rosenbrock'
#cost_name = 'ackley'
#cost_name = 'chasm'

fun, bound = get_cost_function(cost_name)

# Change the solver name to try other algorithms. Warning because genetic algorithm take a population of inital strings
# and not a single one

# solver = AnnealSolver(noisy_step=1, temp_decay=.9, n_iteration=1000) # Load the solver object
# s0 = rd.rand(2) * (bound[1] - bound[0]) + bound[0]  # Define the first coding string candidate randomly
# res = solver.solve(s0, fun, bound)  # Solve the problem

solver = GDFDSolver(learning_rate=.2, exploration_step=0.8, step_decay=.9, n_random_step=9, n_iteration=100)
s0 = rd.rand(2) * (bound[1] - bound[0]) + bound[0]  # Define the first coding string candidate randomly
res = solver.solve(s0, fun, bound)  # Solve the problem

# n_pop = 20
# solver = GeneticSolver(selection_temperature=1, mutation_rate=.03, crossover_rate=.03, n_iteration=50)
# pop0 = [rd.rand(2) * (bound[1] - bound[0]) + bound[0] for k in range(n_pop)]
# res = solver.solve(pop0, fun, bound)  # Solve the problem

# Plot the cost function along iterations
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(res['f_list'],linewidth=2)

ax.set_title('Simulated Annealing')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost value')

# Plot the motion of the solution candidate on the cost function landscape
ax = fig.add_subplot(2, 1, 2, projection='3d')
ax.set_title('Cost function')
plot_cost(ax, fun, bound)
xy = np.array(res['x_list'])
ax.plot(xy[:,0],xy[:,1],res['f_list'][:],color='white',linewidth=2)

plt.show()
