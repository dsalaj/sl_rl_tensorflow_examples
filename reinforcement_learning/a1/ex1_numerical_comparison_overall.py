#!/usr/bin/env python
__author__ = "Guillaume Bellec"

'''
    File name: ex1_numerical_comparisons.py
    Date created: 8/29/2016
    Date last modified: 10/14/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    This file runs the comparison of optimization algorithms on standard non-differential or non-convexe functions.
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

n_random_intilization = 200

score = {'Annealing': {'avg': [], 'dev': []},
         'Gradient': {'avg': [], 'dev': []},
         'Genetic': {'avg': [], 'dev': []}}

for cost_name in ['rastrigin', 'rosenbrock', 'ackley', 'chasm']:
  fun, bound = get_cost_function(cost_name)
  
  # Solve with: Simulated Annealing
  solver = AnnealSolver(noisy_step=1, temp_decay=.9, n_iteration=1000)  # Load the solver object
  
  list_score_anneal = []
  for i in range(n_random_intilization):
      s0 = rd.rand(2) * (bound[1] - bound[0]) + bound[0]  # Define the first solution candidate ramdomly
      res_anneal = solver.solve(s0, fun, bound)  # Solve the problem
      list_score_anneal.append(res_anneal['f_list'][-1])
  
  # Solve with: Gradient Descent
  solver = GDFDSolver(learning_rate=.2, exploration_step=0.8, step_decay=.9, n_random_step=9, n_iteration=100)
  
  list_score_gd = []
  for i in range(n_random_intilization):
      s0 = rd.rand(2) * (bound[1] - bound[0]) + bound[0]  # Define the first solution candidate randomly
      res_gd = solver.solve(s0, fun, bound)  # Solve the problem
      list_score_gd.append(res_gd['f_list'][-1])
  
  # Solve with: Genetic algorithm
  n_pop = 20
  solver = GeneticSolver(selection_temperature=1, mutation_rate=.09, crossover_rate=.09, n_iteration=50)
  
  list_score_genetic = []
  for k in range(n_random_intilization):
      pop0 = [rd.rand(2) * (bound[1] - bound[0]) + bound[0] for k in range(n_pop)]
      res_genetic = solver.solve(pop0, fun, bound)  # Solve the problem
      list_score_genetic.append(res_genetic['f_list'][-1])
  
  
  ##############################
  # Nothing to be modified below
  ##############################
  
  print('FINAL STATISTICS:')
  for name, list_score in zip(['Annealing','Gradient','Genetic'],[list_score_anneal,list_score_gd,list_score_genetic]):
    mean = np.mean(list_score)
    std = np.std(list_score)
    score[name]['avg'].append(mean)
    score[name]['dev'].append(std)
    print('{} \t Averaged cost {:.3g} \t Standard dev. {:.3g}'.format(name,mean,std))
  print('')

print("Final statistics over all cost functions:")
for name in ['Annealing','Gradient','Genetic']:
  temp = score[name]
  mean = np.mean(temp['avg'])
  std = np.mean(temp['dev'])
  print('{} \t Averaged cost {:.3g} \t Standard dev. {:.3g}'.format(name,mean,std))
