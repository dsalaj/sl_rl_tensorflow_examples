#!/usr/bin/env python
__author__ = "Guillaume Bellec"

'''
    File name: ex2_main.py
    Date created: 8/29/2016
    Date last modified: 10/14/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    This file provides an example solution on the CartPole exercise.
    The environment is imported from the gym package.
    Documentation is provided at:
        https://gym.openai.com/

    To install the package on linux run the command:
        pip3 install --user gym

    In this second exercise the students should:
        - Adapt the example to provide a solution with: Simulated Annealing, Gradient Descent and Genetic Algorithm
        - Compare the results and documents the report with numerical results and Figure

    ATTENTION:
        - Unlike in exercise one, the goal is to MAXIMIZE the accumulated reward instead of reducing a cost function

    Follow the instruction of the hand-out
    Feel free to copy this feel and modify it as much as needed.

'''

import gym
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from simulated_annealing import AnnealSolver
from gradient_descent_finite_difference import GDFDSolver
from genetic import GeneticSolver

n_trial_per_call = 10  # Number of trials to estimate the expected accumulated reward at every optimization steps
render = False
render = True  # Run a video of the optimization, set to False TO SAVE COMPUTATION TIME
bound = [-3, 3]  # Bound of the policy parameters [-3,3] is a reasonable choice, no need to change it

# Load the environment
env = gym.make('MountainCar-v0')
n_state = env.observation_space.high.__len__()  # The size of the state space is given by the environment


def policy(parameter, state):
    '''
    The policy is the function that determines the action based on the current observation of the environment.
    This depends on the policy parameters that you should optimize (i.e. the coding strings provided by genetic algorithms).
    For this assignment the policy is defined as a linear classifier:
     - if the scalar product parameter.state is positive: take action 1
     - otherwise take action 0

    :param parameter:
    :param state:
    :return:
    '''

    W1 = parameter[0:2].reshape((1,2))
    W2 = parameter[2:5].reshape((3,1))
    a = np.dot(W1, state)
    y = np.tanh(a)
    out = np.dot(W2, y).reshape((3,))
    res = out.argmax()

    return res


def cost_function(parameter):
    '''
    Fitness function to maximize.
    It opens the environment and runs n_trial_per_call trials for 200 time steps each.
    Nothing to be modified here

    :param parameter:
    :return:
    '''
    list_of_accumulated_rewards = []

    for ep in range(n_trial_per_call):
        observation = env.reset()
        acc_reward = 0
        for t in range(200):  # The number of time steps in this game is maximum 200
            if render: env.render()
            action = policy(parameter, observation)  # Choose an action according to the policy
            observation, reward, done, info = env.step(action)  # Move the CartPole
            acc_reward += reward  # Accumulate the reward
            if done:
                # If the pole falls, we loose.
                break
        list_of_accumulated_rewards.append(acc_reward)

    fitness = sum(list_of_accumulated_rewards)
    fitness /= float(len(list_of_accumulated_rewards))
    cost = (200 - fitness)**2

    return cost


# solver = AnnealSolver(noisy_step=1.5, temp_decay=.99, n_iteration=100, stop_criterion=0)  # Load the solver object
# s0 = np.zeros(5)
# res = solver.solve(s0, cost_function, bound)

# solver = GDFDSolver(learning_rate=.07, exploration_step=.8, step_decay=.97, n_random_step=9, n_iteration=100, stop_criterion=50)
# s0 = np.zeros(5)
# res = solver.solve(s0, cost_function, bound)

n_pop = 20
solver = GeneticSolver(selection_temperature=1, mutation_rate=.13, crossover_rate=.13, n_iteration=50, stop_criterion=0)
pop0 = [rd.rand(5) for k in range(n_pop)]
res = solver.solve(pop0, cost_function, bound)

n_trials = res['n_function_call'] * n_trial_per_call
print('\t Final cost {:.3g} \t in {} trials.'.format(res['f_list'][-1], n_trials))
# -----------

fig, ax = plt.subplots(1)
ax.plot(res['f_list'], lw=2)
ax.set_ylabel('Cost functions')
ax.set_xlabel('Optimization iterations')
plt.show()
