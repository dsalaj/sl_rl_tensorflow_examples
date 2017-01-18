#!/usr/bin/env python

import os
import gym
import time
import copy
import random
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from pong_tools import prepro
from cartpole_utils import plot_results,print_results
from tf_tools import variable_summaries
import tensorflow as tf

# Algorithm parameters
# learning_rate = .01
# gamma = .9
# epsilon = .7
# epsi_decay = .99999
# lr_decay = .99999
# n_hidden = 60

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder



#DQN Paper parameters
epsilon = 0.
learning_rate = 0.00025
lr_decay = 0.99
gamma = 0.95
replay_memory_size = 50000
mb_size = 32



# General parameters
render = True
# render = True
N_print_every = 10
N_trial = 100000000
N_trial_test = 100
# trial_duration = 200

# Generate the environment
env = gym.make("Pong-v0")
# dim_state = env.observation_space.high.__len__()
dim_state = 6  # after preprocessing it's 6
n_action = 3
# action list should be same for all games
# n_action = 3
action_list = [0, 2, 3]  # only 3 controls used
print("Number of valid actions: ", n_action)


img_size = 84
input_ch_num = 2
conv1_filter_size = 8
conv2_filter_size = 4
conv1_feature_maps_num = 16
conv2_feature_maps_num = 32
# conv1_feature_maps_num = 32
# conv2_feature_maps_num = 64
# conv3_feature_maps_num = 64
fc1_size = 256
out_size = n_action
n_hidden = 60

# Generate mini-batches
# Function adjusted from MNIST lasagne tutorial:
# http://lasagne.readthedocs.io/en/latest/user/tutorial.html

class Q_Network():
    def __init__(self, scope):
        with tf.variable_scope(scope):
            # Generate the symbolic variables to hold the state values
            self.state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')
            self.next_state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')

            # Create the parameters of the Q model
            w = tf.Variable(initial_value=tf.truncated_normal([n_hidden, n_action], stddev=0.1), trainable=True, name='weight_variable')
            b = tf.Variable(initial_value=tf.constant(0.1, shape=[n_action]), trainable=True, name='bias')
            wh = tf.Variable(initial_value=tf.truncated_normal([dim_state, n_hidden], stddev=0.1), trainable=True, name='hidden_weight_variable')
            bh = tf.Variable(initial_value=tf.constant(0.1, shape=[n_hidden]), trainable=True, name='hidden_bias')

            # Q function at the current step
            a_y = tf.matmul(self.state_holder, wh, name='output_activation') + bh
            y = tf.nn.relu(a_y, name='hidden_layer_activation')
            a_z = tf.matmul(y, w, name='output_activation') + b

            # Q function at the next step
            next_a_y = tf.matmul(self.next_state_holder, wh, name='next_step_output_activation') + bh
            next_y = tf.nn.relu(next_a_y, name='next_step_hidden_layer_activation')
            next_a_z = tf.matmul(next_y, w, name='next_step_output_activation') + b
            self.next_Q = next_a_z
            self.Q = tf.reshape(a_z, (-1, 1, n_action))
            if scope != 'main':
                self.update_target_network = update_target_graph('main', scope)
            else:
                # Define symbolic variables that will carry information needed for training
                self.action_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_action, 1), name='symbolic_action')
                self.r_holder = tf.placeholder(dtype=tf.float32, shape=(None, 1, 1), name='symbolic_value_estimation')
                self.is_done_holder = tf.placeholder(dtype=tf.float32, shape=(None,), name='is_done')

                # define the role of each training step
                R = tf.batch_matmul(self.Q, self.action_holder)
                # variable_summaries(R, '/R')

                gamma_rew = tf.reshape(gamma * tf.reduce_max(self.next_Q, reduction_indices=1) * (1 - self.is_done_holder), (-1, 1, 1))
                # gamma_rew = tf.reshape(gamma * tf.reduce_max(next_Q, reduction_indices=1)
                #  * (1 - tf.abs(tf.reshape(r_holder,(-1,)))), (-1, 1, 1))
                next_R = self.r_holder + gamma_rew
                # variable_summaries(next_R, '/next_R')

                self.error = tf.clip_by_value(tf.reduce_mean(tf.square((R - next_R))), clip_value_min=-1., clip_value_max=1.)
                variable_summaries(self.error, '/error')

Q_network = Q_Network('main')
target_Q_network = Q_Network('target')

# Define the operation that performs the optimization
# learning_rate_holder = tf.placeholder(dtype=tf.float32, name='symbolic_state')
training_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, momentum=0.0, epsilon=1e-6)\
    .minimize(Q_network.error)
# training_step = tf.train.GradientDescentOptimizer(learning_rate_holder).minimize(error)

sess = tf.Session()  # FOR NOW everything is symbolic, this object has to be called to compute each value of Q
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state("saved_networks")
if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded weights from:", checkpoint.model_checkpoint_path)
else:
        print("Weights were initialised randomly.")

suffix = time.strftime('%Y-%m-%d--%H-%M-%S')
train_writer = tf.train.SummaryWriter('tensorboard/cartpole/{}'.format(suffix) + '/train', sess.graph)
merged = tf.merge_all_summaries()


def policy(state):
    """
    epsilon greedy policy:
    - with probability epsilon take a random action
    - otherwise take an action the action that maximizes Q(s,a) with s the current state

    :param Q_table:
    :param state:
    :return:
    """
    if rd.rand() < epsilon:
        return rd.choice(n_action)


    Q_values = sess.run(target_Q_network.Q,
                        feed_dict={target_Q_network.state_holder: state.reshape(1, dim_state)})
    print("-------------")
    print(Q_values)
    print("-------------")

    val = np.max(Q_values[0, :])
    max_indices = np.where(Q_values[0, 0, :] == val)[0]
    a = rd.choice(max_indices)
    return a



time_list = []
reward_list = []
err_list = []
val_list = []
exp_mem = []
skip_counter = 0
frames_processed = 0
summary_counter = 0
descent_steps = 0
for k in range(N_trial + N_trial_test):
    if k % 10000 == 0:
	    print("Current iter k: ", k)
    if k > N_trial:
        epsilon = 0
        learning_rate = 0

    acc_reward = 0  # Init the accumulated reward
    obs1 = obs2 = obs3 = obs4 = env.reset()  # Init the state
    # pro_observation = prepro(obs1, obs2, obs3, obs4)
    pro_observation = prepro(obs1, obs4)
    action = policy(pro_observation)  # Init the first action

    trial_err_list = []

    t = 0
    point_length = 0
    # for t in range(trial_duration):
    while True:
        if render: env.render()

        obs4, reward, done, info = env.step(action_list[action])  # Take the action

        pro_new_observation = prepro(obs1, obs4)


        point_length += 1

        if(reward != 0):
            print("Point length: ", point_length)
            point_length = 0


                # exp_mem = []

        one_hot_action = np.zeros((1, n_action))
        one_hot_action[0, action] = 1.
        # Compute the Bellman Error for monitoring
        # err = sess.run(error, feed_dict={
        #     state_holder: pro_observation.reshape((-1,img_size,img_size,input_ch_num)),
        #     next_state_holder: pro_new_observation.reshape((-1,img_size,img_size,input_ch_num)),
        #     action_holder: one_hot_action.reshape(-1, n_action, 1),
        #     is_done_holder: np.array(done).reshape(-1,),
        #     r_holder: np.array(reward).reshape(-1, 1, 1)})
        # Add the error in a trial-specific list of errors
        # trial_err_list.append(err)

        # Pass the new state to the next step
        obs1 = obs2
        obs2 = obs3
        obs3 = obs4
        pro_observation = pro_new_observation  # Pass the new state to the next step
        action = policy(pro_observation)  # Decide next action based on policy
        acc_reward += reward  # Accumulate the reward

        t += 1

        if done: break

    # Stack values for monitoring
    err_list.append(np.mean(trial_err_list))
    time_list.append(t + 1)
    reward_list.append(acc_reward)  # Store the result

    # save network every 100000 iteration

    print_results(k, time_list, err_list, reward_list, N_print_every=N_print_every)

plot_results(N_trial, N_trial_test, reward_list, time_list, err_list)
plt.show()
