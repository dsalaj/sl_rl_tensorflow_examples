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
gamma = .7
epsilon = .0
render = False
N_trial = 1000
N_trial_test = 100
trial_duration = 100


# Generate the environment
env = FrozenLakeEnv(map_name='8x8', is_slippery=False)
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
        a = rd.randint(low=0, high=n_action)
    else:
        max_actions = np.argwhere(Q_table[state] == np.amax(Q_table[state])).flatten()
        a = rd.choice(max_actions)
    return a


def update_Q_table(Q_table, e, state, action, reward, new_state, new_action, is_done):
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
        d = (reward + gamma * Q_table[new_state, new_action] - Q_table[state, action])

    # reference: http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node77.html
    e[state, action] += 1
    l = 0.5  # lambda

    for s in range(Q_table.shape[0]):
        for a in range(Q_table.shape[1]):
            Q_table[s, a] += learning_rate * d * e[s, a]
            e[s, a] *= gamma * l


reward_list = []
for k in range(N_trial + N_trial_test):

    # Initialize the eligibility traces
    e = np.zeros((n_state, n_action))

    acc_reward = 0  # Init the accumulated reward
    observation = env.reset()  # Init the state
    action = policy(Q_table, observation, epsilon)  # Init the first action

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

        new_observation, reward, done, info = env.step(action)  # Take the action
        new_action = policy(Q_table, new_observation, epsilon)
        update_Q_table(Q_table, e, observation, action, reward, new_observation, new_action, done)

        # DEBUG plots
        if reward > 0 and t > 5 and False:
            start_position = np.concatenate(np.where(env.desc == b'S'))
            goal_position = np.concatenate(np.where(env.desc == b'G'))
            fig, ax_list = plt.subplots(1, 1)
            V = np.max(Q_table, axis=1).reshape(env.ncol, env.nrow)
            im = ax_list.imshow(V, interpolation='nearest', vmin=np.floor(V.min()), vmax=np.ceil(V.max()))
            ax_list.set_xticks([])
            ax_list.set_yticks([])
            ax_list.set_title('Value fun. (from Q table)')
            ax_list.annotate('S', xy=(start_position[0]-.15, start_position[0]+.15), color='white', size=20)
            ax_list.annotate('G', xy=(goal_position[0]-.15, goal_position[0]+.15), color='white', size=20)

            cbar_ax = fig.add_axes([0.925, 0.05, 0.025, 0.3])
            fig.colorbar(im, cax=cbar_ax, ticks=[np.floor(V.min()), np.ceil(V.max())])
            plt.show()

        #####################

        observation = new_observation  # Pass the new state to the next step
        action = new_action  # Pass the new action to the next step
        acc_reward += reward  # Accumulate the reward
        if done:
            break  # Stop the trial when you fall in a hole or when you find the goal

    reward_list.append(acc_reward)  # Store the result

print('Average accumulated reward in {} test runs: {:.3g}'.format(N_trial_test,np.mean(reward_list[N_trial:])))
plot_results(N_trial, N_trial_test, reward_list, Q_table, env)
plt.show()
