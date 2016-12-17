#!/usr/bin/env python
__author__ = "Guillaume Bellec"

'''
    File name: ex4_cartepole_non_lin_sarsa.py
    Date created: 10/29/2016
    Date last modified: 11/09/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    This file aims at solving the CartPole environment from gym (gym.openai.org) with Tensorflow (tensorflow.org).
    The student should implement the on-policy version of Q-Learning (SARSA) by modeling the Q function as a neural network.


'''

import gym
import copy
import random
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from gym.envs.classic_control.cartpole import CartPoleEnv
from cartpole_utils import plot_results, print_results
import tensorflow as tf

# Algorithm parameters
learning_rate = 3e-1
gamma = .9
epsilon = 1.
epsi_decay = .999
lr_decay = .999
n_hidden = 20

# General parameters
render = False
N_print_every = 100
N_trial = 2000
N_trial_test = 100
trial_duration = 200

# Generate the environment
env = CartPoleEnv()
dim_state = env.observation_space.high.__len__()
n_action = env.action_space.n

# Initialize the parameters of the Q model
wh0 = np.float32(rd.normal(loc=0.0, scale=1./np.sqrt(dim_state), size=(dim_state, n_hidden)))
bh0 = np.zeros(n_hidden, dtype=np.float32)
w0 = np.float32(rd.normal(loc=0.0, scale=1./np.sqrt(n_hidden), size=(n_hidden, n_action)))
b0 = np.zeros(n_action, dtype=np.float32)

# Generate the symbolic variables to hold the state values
state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')
next_state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')

# Create the parameters of the Q model
w = tf.Variable(initial_value=w0, trainable=True, name='weight_variable')
b = tf.Variable(initial_value=b0, trainable=True, name='bias')
wh = tf.Variable(initial_value=wh0, trainable=True, name='hidden_weight_variable')
bh = tf.Variable(initial_value=bh0, trainable=True, name='hidden_bias')

# Q function at the current step
a_y = tf.matmul(state_holder, wh, name='output_activation') + bh
y = tf.nn.relu(a_y, name='hidden_layer_activation')
print("hidden layer 1 shape", y.get_shape())
a_z = tf.matmul(y, w, name='output_activation') + b
Q = - tf.sigmoid(a_z, name='Q_model')
print("Q shape", Q.get_shape())

# Q function at the next step
next_a_y = tf.matmul(next_state_holder, wh, name='next_step_output_activation') + bh
next_y = tf.nn.relu(next_a_y, name='next_step_hidden_layer_activation')
print("hidden layer 2 shape", next_y.get_shape())
next_a_z = tf.matmul(next_y, w, name='next_step_output_activation') + b
next_Q = - tf.nn.sigmoid(next_a_z, name='next_step_Q_model')
print("next_Q shape", next_Q.get_shape())

# Define symbolic variables that will carry information needed for training
action_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_action, 1), name='symbolic_action')
print("action_holder shape", action_holder.get_shape())
# next_action_holder = tf.placeholder(dtype=tf.int32, name='symbolic_next_action')
r_holder = tf.placeholder(dtype=tf.float32, shape=(None,), name='symbolic_value_estimation')
is_done_holder = tf.placeholder(dtype=tf.float32, shape=(None,), name='is_done')

# define the role of each training step
# R = Q[:, 0, action_holder[:]]
Q = tf.reshape(Q, (-1, 1, n_action))
print("Q shape", Q.get_shape())
# next_Q = tf.reshape(next_Q, (-1, 1, n_action))
# print("next_Q shape", next_Q.get_shape())

R = tf.batch_matmul(Q, action_holder)
print("R shape", R.get_shape())
# R = tf.reduce_sum(r, reduction_indices=0)
# print("R shape", R.get_shape())
rm = tf.reduce_max(next_Q, reduction_indices=1)
print("rm shape", rm.get_shape())
r_ = gamma * rm * (1 - is_done_holder)
print("r_ shape", r_.get_shape())
next_R = r_holder + r_
print("next_R shape", next_R.get_shape())

diff = (tf.reshape(R, (-1,)) - next_R)**2
print("diff shape", diff.get_shape())
error = tf.reduce_mean(diff)
print("error shape", error.get_shape())

# Define the operation that performs the optimization
learning_rate_holder = tf.placeholder(dtype=tf.float32, name='symbolic_state')
training_step = tf.train.GradientDescentOptimizer(learning_rate_holder).minimize(error)

sess = tf.Session()  # FOR NOW everything is symbolic, this object has to be called to compute each value of Q
sess.run(tf.initialize_all_variables())


def policy(state):
    '''
    This should implement an epsilon greedy policy:
    - with probability epsilon take a random action
    - otherwise take an action the action that maximizes Q(s,a) with s the current state

    :param Q_table:
    :param state:
    :return:
    '''
    if rd.rand() < epsilon:
        return rd.randint(0, n_action)

    Q_values = sess.run(Q, feed_dict={state_holder: state.reshape(1, dim_state)})
    val = np.max(Q_values[0, :])
    max_indices = np.where(Q_values[0, :] == val)[0]
    return rd.choice(max_indices)


def predict(state):

    Q_values = sess.run(Q, feed_dict={state_holder: state.reshape(1, dim_state)})
    val = np.max(Q_values[0, :])
    max_indices = np.where(Q_values[0, :] == val)[0]
    return rd.choice(max_indices)


time_list = []
reward_list = []
err_list = []
val_list = []
memory_pool = []

for k in range(N_trial + N_trial_test):
    epsilon *= epsi_decay
    learning_rate *= lr_decay

    if k > N_trial:
        epsilon = 0
        learning_rate = 0

    # Init the accumulated reward, and state of action pair of previous state
    acc_reward = 0  # Init the accumulated reward
    observation = env.reset()  # Init the state
    action = policy(observation)  # Init the first action

    trial_err_list = []

    for t in range(trial_duration):  # The number of time steps in this game is maximum 200
        if render: env.render()

        new_observation, reward, done, info = env.step(action)  # Take the action
        reward = 0
        if done and t < 199: reward = -1    # The reward is modified

        memory_pool.append(zip((observation, new_observation, action, reward, done)))
        # print(observation, new_observation, action, reward, done)
        # print(memory_pool)
        # mb_obs, mb_nob, mb_act, mb_rew, mb_don = zip(*(memory_pool))
        # print(mb_obs, mb_nob, mb_act, mb_rew, mb_don)
        # print(t, len(memory_pool))
        new_action = predict(new_observation)  # Compute the next action

        mb_size = 1
        if len(memory_pool) > 0 and len(memory_pool) % mb_size == 0:
            # minibatch_zip_ = random.sample(memory_pool, mb_size)
            minibatch_zip_ = [memory_pool[-1],]
            minibatch_zip = copy.deepcopy(minibatch_zip_)
            # print("LEARNING", len(minibatch_zip))
            mb_obs, mb_nob, mb_act, mb_rew, mb_don = zip(*minibatch_zip)
            # print(mb_obs, mb_nob, mb_act, mb_rew, mb_don)

            mb_obs = np.array(mb_obs)  # [o  for o, no, a, r, d in minibatch_zip])
            mb_nob = np.array(mb_nob)  # [no for o, no, a, r, d in minibatch_zip])
            mb_act = np.array(mb_act)  # [a  for o, no, a, r, d in minibatch_zip])
            mb_rew = np.array(mb_rew)  # [r  for o, no, a, r, d in minibatch_zip])
            mb_don = np.array(mb_don).astype(np.float32)  # [d  for o, no, a, r, d in minibatch_zip])
            one_hot_actions = np.zeros((mb_size, n_action))
            one_hot_actions[np.arange(mb_size), mb_act] = 1

            # Perform one step of gradient descent
            sess.run(training_step, feed_dict={
                state_holder: mb_obs.reshape(-1, dim_state),
                next_state_holder: mb_nob.reshape(-1, dim_state),
                action_holder: one_hot_actions.reshape(-1, n_action, 1),
                # next_action_holder: new_action,
                is_done_holder: mb_don.reshape(-1,),
                r_holder: mb_rew.reshape(-1,),
                learning_rate_holder: learning_rate})

            # Compute the Bellman Error for monitoring
            err = sess.run(error, feed_dict={
                state_holder: mb_obs.reshape(-1, dim_state),
                next_state_holder: mb_nob.reshape(-1, dim_state),
                action_holder: one_hot_actions.reshape(-1, n_action, 1),
                # next_action_holder: new_action,
                is_done_holder: mb_don.reshape(-1,),
                r_holder: mb_rew.reshape(-1,)})

            # Add the error in a trial-specific list of errors
            trial_err_list.append(err)

        observation = new_observation  # Pass the new state to the next step
        # action = new_action
        action = policy(observation)  # Decide next action based on policy
        acc_reward += reward  # Accumulate the reward

        if done:
            break  # Stop the trial when the environment says it is done

    # Stack values for monitoring
    err_list.append(np.mean(trial_err_list))
    time_list.append(t + 1)
    reward_list.append(acc_reward)  # Store the result

    print_results(k, time_list, err_list, reward_list)

plot_results(N_trial,N_trial_test,reward_list,time_list, err_list)
plt.show()
