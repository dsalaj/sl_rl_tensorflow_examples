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
import time
import copy
import random
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from gym.envs.classic_control.cartpole import CartPoleEnv
from cartpole_utils import plot_results,print_results
from tf_tools import variable_summaries
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
print("y shape", y.get_shape())
a_z = tf.matmul(y, w, name='output_activation') + b
Q = - tf.sigmoid(a_z, name='Q_model')
print("Q shape", Q.get_shape())

# Q function at the next step
next_a_y = tf.matmul(next_state_holder, wh, name='next_step_output_activation') + bh
next_y = tf.nn.relu(next_a_y, name='next_step_hidden_layer_activation')
print("next_y shape", next_y.get_shape())
next_a_z = tf.matmul(next_y, w, name='next_step_output_activation') + b
next_Q = - tf.nn.sigmoid(next_a_z, name='next_step_Q_model')
print("next_Q shape", next_Q.get_shape())

# Define symbolic variables that will carry information needed for training
action_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_action, 1), name='symbolic_action')
# action_holder = tf.placeholder(dtype=tf.int32, name='symbolic_action')
r_holder = tf.placeholder(dtype=tf.float32, name='symbolic_value_estimation')
is_done_holder = tf.placeholder(dtype=tf.float32, name='is_done')

# define the role of each training step
Q = tf.reshape(Q, (-1, 1, n_action))
print("Q shape", Q.get_shape())
# # print("action_holder shape", action_holder.get_shape())
# R = tf.matmul(Q, action_holder)
R = tf.batch_matmul(Q, action_holder)
# R = Q[0, action_holder[0]]
# variable_summaries(R, '/R')
print("R shape", R.get_shape())
next_R = r_holder + gamma * tf.reduce_max(next_Q, reduction_indices=1) * (1 - is_done_holder)
# variable_summaries(next_R, '/next_R')
print("next_R shape", next_R.get_shape())

error = (R - next_R)**2
variable_summaries(error, '/error')
print("error shape", error.get_shape())

# Define the operation that performs the optimization
learning_rate_holder = tf.placeholder(dtype=tf.float32, name='symbolic_state')
training_step = tf.train.GradientDescentOptimizer(learning_rate_holder).minimize(error)

sess = tf.Session()  # FOR NOW everything is symbolic, this object has to be called to compute each value of Q
sess.run(tf.initialize_all_variables())

suffix = time.strftime('%Y-%m-%d--%H-%M-%S')
train_writer = tf.train.SummaryWriter('tensorboard/cartpole/{}'.format(suffix) + '/train', sess.graph)
merged = tf.merge_all_summaries()


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
    max_indices = np.where(Q_values[0, 0, :] == val)[0]
    # print("---------- policy")
    # print("Q_values shape", Q_values.shape)
    # print("Q_values", Q_values)
    # print("val", val)
    # print("max_indices", max_indices)
    return rd.choice(max_indices)


# def predict(state):
#
#     Q_values = sess.run(Q, feed_dict={state_holder: state.reshape(1, dim_state)})
#     val = np.max(Q_values[0, :])
#     max_indices = np.where(Q_values[0, :] == val)[0]
#     return rd.choice(max_indices)


time_list = []
reward_list = []
err_list = []
val_list = []
exp_mem = []

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

        exp_mem.append(zip((observation, new_observation, action, reward, done)))
        mb_size = 2

        if t > 1:
            # minibatch_zip_ = random.sample(exp_mem, mb_size)
            minibatch_zip_ = [exp_mem[-1], exp_mem[-2]]
            minibatch_zip = copy.deepcopy(minibatch_zip_)
            mb_obs, mb_nob, mb_act, mb_rew, mb_don = zip(*minibatch_zip)
            assert((observation == mb_obs[0][0]).all())
            assert((new_observation == mb_nob[0][0]).all())
            assert(action == mb_act[0][0])
            assert(reward == mb_rew[0][0])
            assert(done == mb_don[0][0])

            mb_obs = np.array(mb_obs)  # [o  for o, no, a, r, d in minibatch_zip])
            mb_nob = np.array(mb_nob)  # [no for o, no, a, r, d in minibatch_zip])
            mb_act = np.array(mb_act)  # [a  for o, no, a, r, d in minibatch_zip])
            mb_rew = np.array(mb_rew)  # [r  for o, no, a, r, d in minibatch_zip])
            mb_don = np.array(mb_don).astype(np.float32)  # [d  for o, no, a, r, d in minibatch_zip])
            one_hot_actions = np.zeros((mb_size, n_action))
            one_hot_actions[np.arange(mb_size), np.reshape(mb_act, (mb_size,))[:]] = 1
            # print("------------")
            # print("mb_act", mb_act)
            # print("one_hot_actions", one_hot_actions)
            # print("one_hot_actions shape", one_hot_actions.shape)

            # Perform one step of gradient descent
            summary, _ = sess.run([merged, training_step], feed_dict={
                state_holder: mb_obs.reshape(-1, dim_state),
                next_state_holder: mb_nob.reshape(-1, dim_state),
                # action_holder: mb_act.reshape(-1,),
                action_holder: one_hot_actions.reshape(-1, n_action, 1),
                is_done_holder: mb_don.reshape(-1,),
                r_holder: mb_rew.reshape(-1,),
                learning_rate_holder: learning_rate})
            train_writer.add_summary(summary, k*trial_duration + t)

            # Compute the Bellman Error for monitoring
            err = sess.run(error, feed_dict={
                state_holder: mb_obs.reshape(-1, dim_state),
                next_state_holder: mb_nob.reshape(-1, dim_state),
                # action_holder: mb_act.reshape(-1,),
                action_holder: one_hot_actions.reshape(-1, n_action, 1),
                is_done_holder: mb_don.reshape(-1,),
                r_holder: mb_rew.reshape(-1,)})
            # Add the error in a trial-specific list of errors
            trial_err_list.append(err)

        # new_action = predict(new_observation)  # Compute the next action

        observation = new_observation  # Pass the new state to the next step
        # action = new_action
        action = policy(observation)  # Decide next action based on policy
        acc_reward += reward  # Accumulate the reward


        if done:
            break  # Stop the trial when the environment says it is done

    # summary = sess.run(merged, feed_dict={
    #     state_holder: mb_obs.reshape(-1, dim_state),
    #     next_state_holder: mb_nob.reshape(-1, dim_state),
    #     action_holder: mb_act.reshape(-1,),
    #     # action_holder: one_hot_actions.reshape(-1, n_action, 1),
    #     is_done_holder: mb_don.reshape(-1,),
    #     r_holder: mb_rew.reshape(-1,),
    #     learning_rate_holder: learning_rate})
    # train_writer.add_summary(summary, k)

    # Stack values for monitoring
    err_list.append(np.mean(trial_err_list))
    time_list.append(t + 1)
    reward_list.append(acc_reward)  # Store the result

    print_results(k, time_list, err_list, reward_list)

plot_results(N_trial,N_trial_test,reward_list,time_list, err_list)
plt.show()
