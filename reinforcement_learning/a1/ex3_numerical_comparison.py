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
# render = True  # Run a video of the optimization, set to False TO SAVE COMPUTATION TIME
bound = [-3, 3]  # Bound of the policy parameters [-3,3] is a reasonable choice, no need to change it

# Load the environment
env = gym.make('MountainCar-v0')
n_state = env.observation_space.high.__len__()  # The size of the state space is given by the environment




#Policy without hidden layer
# def policy(parameter, state):
#     '''
#     The policy is the function that determines the action based on the current observation of the environment.
#     This depends on the policy parameters that you should optimize (i.e. the coding strings provided by genetic algorithms).
#     For this assignment the policy is defined as a linear classifier:
#      - if the scalar product parameter.state is positive: take action 1
#      - otherwise take action 0
#
#     :param parameter:
#     :param state:
#     :return:
#     '''
#     #print state
#     res = np.dot(parameter, state) > 0
#     if res == 1 : res = 2
#
#     return res

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
    #print state
    W1 = parameter[:6].reshape((3,2))
    W2 = parameter[6:9]
    a = np.dot(W1,state)
    y = np.tanh(a)
    res = int(np.dot(W2,y) > 0)
    if res == 1 : res = 2

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
    succ_counter = 0
    for ep in range(n_trial_per_call):
        observation = env.reset()
        acc_reward = 0
        last_action = 0
        min_x = observation[0]
        max_x = observation[0]
        steps_reward = 0
        for t in range(200):  # The number of time steps in this game is maximum 200
            if render: env.render()
            action = policy(parameter, observation)  # Choose an action according to the policy
            observation, reward, done, info = env.step(action)  # Move the Mountain car
            if(observation[0] <= min_x):
                min_x = observation[0]
            else:
                max_x = observation[0]        #print acc_reward
            steps_reward += reward  # Accumulate the reward
            if done:
                # If the car reaches flag, we win.
                break
        if steps_reward > -200:
            # give bonus reward for completion
            acc_reward = 2
            # count success for succession rate
            succ_counter += 1
        # We measure the score by the distance covered in
        # simulation
        acc_reward += np.abs(min_x-max_x)
        #print acc_reward
        list_of_accumulated_rewards.append(acc_reward)

    fitness = sum(list_of_accumulated_rewards)
    fitness /= float(len(list_of_accumulated_rewards))

    # Cost function is set to 0 to stop the algorithm
    # if simulation completes the task for n_trial_per_call
    cost = 1/(fitness)**2
    if(succ_counter == n_trial_per_call):
        cost = 0
    return cost


n_random_intilization = 100
#
# solver = AnnealSolver(noisy_step=2.5, temp_decay=.94, n_iteration=100, stop_criterion=0)  # Load the solver object
# list_score_anneal = []
# for i in range(n_random_intilization):
#     s0 = rd.rand(n_state) * (bound[1] - bound[0]) + bound[0]  # Define the first solution candidate ramdomly
#     res_anneal = solver.solve(s0, cost_function, bound)
#     if res_anneal['f_list'][-1] == 0:
#         list_score_anneal.append(res_anneal['n_function_call'])
#
# solver = GDFDSolver(learning_rate=.07, exploration_step=.8, step_decay=.99, n_random_step=9, n_iteration=100, stop_criterion=0)
# list_score_gd = []
# for i in range(n_random_intilization):
#     s0 = rd.rand(n_state) * (bound[1] - bound[0]) + bound[0]  # Define the first solution candidate randomly
#     res_gd = solver.solve(s0, cost_function, bound)  # Solve the problem
#     if res_gd['f_list'][-1] == 0:
#         list_score_gd.append(res_gd['n_function_call'])

n_pop = 20
solver = GeneticSolver(selection_temperature=1, mutation_rate=.03, crossover_rate=.03, n_iteration=50, stop_criterion=0)
list_score_genetic = []
for k in range(n_random_intilization):
    pop0 = [rd.rand(n_state) * (bound[1] - bound[0]) + bound[0] for k in range(n_pop)]
    res_genetic = solver.solve(pop0, cost_function, bound)  # Solve the problem
    if res_genetic['f_list'][-1] == 0:
        list_score_genetic.append(res_genetic['n_function_call'])

##############################
# Nothing to be modified below
##############################

print('\n FINAL STATISTICS:')
for name, list_score in zip(['Genetic'],[list_score_genetic]):

    mean = np.mean(list_score)
    std = np.std(list_score)

    print('{} \t Averaged number of function calls {:.3g} \t Standard dev. {:.3g}'.format(name,mean,std))


# s0 = np.asarray([
# -0.845226402928 ,
# 2.40399032615 ,
# 0.322772761635 ,
# 0.643287316438 ,
# 0.605076847906 ,
# 0.0482834095796 ,
# 0.482324106981 ,
# 0.668424652057 ,
# 0.279595699179 ,
# ]
#
# )