#!/usr/bin/env python
__author__ = "Guillaume Bellec"

'''
    File name: ex1_three_algorithms.py
    Date created: 8/29/2016
    Date last modified: 10/14/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    This file runs the comparison of optimization algorithms on standard non-differential or non-convexe functions.
    Follow the instructions in the hand-out.

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
# cost_name = 'ackley'
# cost_name = 'chasm'

fun, bound = get_cost_function(cost_name)

# Solve with: Simulated Annealing
solver = AnnealSolver(noisy_step=1, temp_decay=.99, n_iteration=1000)  # Load the solver object
s0 = rd.rand(2) * (bound[1] - bound[0]) + bound[0]  # Define the first solution candidate ramdomly
res_anneal = solver.solve(s0, fun, bound)  # Solve the problem

# Solve with: Gradient Descent
solver = GDFDSolver(learning_rate=1, exploration_step=1, step_decay= .9, n_random_step=9, n_iteration=100)
res_gd = solver.solve(s0, fun, bound)  # Solve the problem

# Solve with: Genetic algorithm
n_pop = 20
solver = GeneticSolver(selection_temperature=1, mutation_rate=.03, crossover_rate=.03, n_iteration=50)
pop0 = [rd.rand(2) * (bound[1] - bound[0]) + bound[0] for k in range(n_pop)]
res_genetic = solver.solve(pop0, fun, bound)  # Solve the problem


##############################
# Nothing to be modified below
##############################

# Plot the cost function along iterations
fig = plt.figure()
f_min, f_max = np.Inf, -np.Inf
for k, res, name in zip(range(2), [res_anneal, res_gd], ['Simulated Annealing', 'Gradient Descent']):
    # For the result of each algorithm do the following plot
    ax = fig.add_subplot(2, 3, 1 + k)
    ax.plot(res['f_list'], linewidth=2)

    ax.set_title(name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost value')

    # Plot the motion of the solution candidate on the cost function landscape
    ax = fig.add_subplot(2, 3, 4 + k, projection='3d')
    ax.set_title('Cost function')
    plot_cost(ax, fun, bound)
    xy = np.array(res['x_list'])
    ax.plot(xy[:, 0], xy[:, 1], res['f_list'][:], color='white', linewidth=2)

    f_min = min(f_min, np.min(res['f_list']))
    f_max = max(f_max, np.max(res['f_list']))

# Do a specific plot for the genetic algorithm to show the population of policy parameter candidates
k, res, name = 2, res_genetic, 'Genetic'
# For the result of each algorithm do the following plot
ax = fig.add_subplot(2, 3, 1 + k)
for i in range(5):
    ax.plot(res['f_pop_list'][:, i], linewidth=2, color='gray', alpha=.5)
ax.plot(res['f_list'], linewidth=2, color='blue')
ax.set_title(name)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost value')

# Plot the motion of the solution candidate on the cost function landscape
ax = fig.add_subplot(2, 3, 4 + k, projection='3d')
ax.set_title('Cost function')
plot_cost(ax, fun, bound)

for i in range(5):
    xy = res['pop_list'][:, i, :]
    ax.plot(xy[:, 0], xy[:, 1], res['f_pop_list'][:, i], color='gray', linewidth=2)

xy = np.array(res['x_list'])
ax.plot(xy[:, 0], xy[:, 1], res['f_list'][:], color='white', linewidth=2)

f_min = min(f_min, np.min(res['f_list']))
f_max = max(f_max, np.max(res['f_list']))

for k in range(3):
    ax = fig.add_subplot(2, 3, 1 + k)
    ax.set_ylim([f_min, f_max])

plt.show()
