#!/usr/bin/env python
__author__ = "Guillaume Bellec"

'''
    File name: ex3_deterministic.py
    Date created: 10/29/2016
    Date last modified: 11/02/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    This file aims at solving the FrozenLake environment from gym (gym.openai.org).
    In the deterministic case, the floor in not slippery and this example becomes a basic grid world environment.

    The student should implement SARSA and Q-learning to solve this problem.


'''

import gym
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from frozenlake_utils import plot_results

# Algorithm parameters
learning_rate = 0.5
gamma = 1.
epsilon = .01
render = False
N_trial = 10000
N_trial_test = 100
trial_duration = 100


# Generate the environment
env = FrozenLakeEnv(map_name='4x4', is_slippery=True)
n_state = env.observation_space.n
n_action = env.action_space.n

# Initialize the Q values
Q_table = np.zeros((n_state, n_action))


def policy(Q_table, state, epsilon):
    '''
    This should implement an epsilon greedy policy.

    :param Q_table:
    :param state:
    :return:
    '''

    # TODO: Implement the epsilon greedy policy
    # - with probability epsilon return a random action
    # - otherwise for with s the current state:
    #       find the actions that maximizes Q(s,a)
    #       return randomly one of them

    if rd.uniform() < epsilon:
        return rd.randint(low=0, high=n_action)
    else:
        actions = np.argwhere(Q_table[state] == np.amax(Q_table[state])).flatten()
        return rd.choice(actions)


def update_Q_table(Q_table, state, action, reward, new_state, is_done):
    '''
    Update the Q values according to the SARSA algorithm.

    :param Q_table: Table of expected return for each pair of action and state
    :param state:
    :param action:
    :param reward:
    :param new_state:
    :param new_action:
    :param is_done:
    :return:
    '''

    # TODO: Implement the update of the Q table required in the SARSA algorithm
    # WARNING: If the trial stops, make sure to set 0 as the estimation of all future return

    if is_done:
        d = (reward - Q_table[state, action])
        # FIXME: new_state Q should be set to 0
    else:
        d = (reward + gamma * np.max(Q_table[new_state, :]) - Q_table[state, action])

    Q_table[state, action] += learning_rate * gamma * d




reward_list = []
for k in range(N_trial + N_trial_test):

    acc_reward = 0  # Init the accumulated reward
    observation = env.reset()  # Init the state

    for t in range(trial_duration):  # The number of time steps in this game is maximum 200
        if render: env.render()

        #####################
        # TODO:
        # In the APPROPRIATE order do the following:
        #   - Update the Q table with the function update_Q_table()
        #   - choose an action with the function policy()
        #   - perform an action
        #
        # TIP:
        #   Think about the final state the specific case just before getting the reward for the first time.
        #   The "goodness" of the rewarding state has to propagate to the previous one.

        action = policy(Q_table, observation, epsilon)  # Init the first action
        new_observation, reward, done, info = env.step(action)  # Take the action

        if reward == 1 and done:
            modified_reward = 100
        elif reward == 0 and done:
            modified_reward = -100
        else:
            modified_reward = -1

        update_Q_table(Q_table, observation, action, modified_reward, new_observation, done)

        #####################

        observation = new_observation  # Pass the new state to the next step
        acc_reward += reward  # Accumulate the reward
        if done:
            break  # Stop the trial when you fall in a hole or when you find the goal

    reward_list.append(acc_reward)  # Store the result

print('Average accumulated reward in {} test runs: {:.3g}'.format(N_trial_test,np.mean(reward_list[N_trial:])))
plot_results(N_trial, N_trial_test, reward_list, Q_table, env)
plt.show()
