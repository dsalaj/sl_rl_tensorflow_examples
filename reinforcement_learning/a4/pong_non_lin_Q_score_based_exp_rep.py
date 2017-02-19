#!/usr/bin/env python

import gym
import time
import copy
import random
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from gym.envs.classic_control.cartpole import CartPoleEnv
from pong_tools import prepro
from cartpole_utils import plot_results,print_results
from tf_tools import variable_summaries
import tensorflow as tf

# # Algorithm parameters
# learning_rate = 0.0001
# gamma = .999
# epsilon = .2
# epsi_decay = 1#.9999
# lr_decay = 1# .9999
# n_hidden = 60

# Minimal bellman error reached 0.0112

# Algorithm parameters
learning_rate = .005
gamma = .99
epsilon = .5
epsilon_min = .1
epsi_decay = .99
lr_decay = .9999
n_hidden = 50
exploration_period = 100
exploration_steps = 10000
exp_pos = 0
exp_neg = 0

# General parameters
render = False
# render = True
N_print_every = 10
N_trial = 10000
N_trial_test = 100
trial_duration = 200

# Generate the environment
env = gym.make("Pong-v0")
# dim_state = env.observation_space.high.__len__()
dim_state = 6  # after preprocessing it's 6
# n_action = env.action_space.n
n_action = 3
action_list = [0, 2, 3]


def neg_prop(mb_rew_sum, i, batch_len):
    if i > batch_len - 10:  # punish only last 10 frames
        return (mb_rew_sum * ((i+1) / batch_len)**4,)
    else:
        return (0, )


def iterate_minibatches(exp_mem, shuffle=False, batchsize=32):
    if shuffle:
        indices = np.arange(len(exp_mem))
    np.random.shuffle(indices)
    for start_idx in range(0, len(exp_mem) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        exp_mem_yield = [exp_mem[i] for i in excerpt]
        yield exp_mem_yield


# Generate the symbolic variables to hold the state values
state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')
next_state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')

# Create the parameters of the Q model - initializing with non-zero values
w = tf.Variable(initial_value=tf.truncated_normal([n_hidden, n_action], stddev=0.1), trainable=True, name='weight_variable')
b = tf.Variable(initial_value=tf.constant(0.1, shape=[n_action]), trainable=True, name='bias')
wh = tf.Variable(initial_value=tf.truncated_normal([dim_state, n_hidden], stddev=0.1), trainable=True, name='hidden_weight_variable')
bh = tf.Variable(initial_value=tf.constant(0.1, shape=[n_hidden]), trainable=True, name='hidden_bias')

# Q function at the current step
a_y = tf.matmul(state_holder, wh, name='output_activation') + bh
y = tf.nn.relu(a_y, name='hidden_layer_activation')
a_z = tf.matmul(y, w, name='output_activation') + b
# Q = -tf.nn.softmax(a_z, name='Q_model')
Q = a_z

# Q function at the next step
next_a_y = tf.matmul(next_state_holder, wh, name='next_step_output_activation') + bh
next_y = tf.nn.relu(next_a_y, name='next_step_hidden_layer_activation')
next_a_z = tf.matmul(next_y, w, name='next_step_output_activation') + b
# next_Q = -tf.nn.softmax(next_a_z, name='next_step_Q_model')
next_Q = next_a_z

# Define symbolic variables that will carry information needed for training
action_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_action, 1), name='symbolic_action')
r_holder = tf.placeholder(dtype=tf.float32, shape=(None, 1, 1), name='symbolic_value_estimation')
is_done_holder = tf.placeholder(dtype=tf.float32, shape=(None,), name='is_done')

# define the role of each training step
Q = tf.reshape(Q, (-1, 1, n_action))
R = tf.batch_matmul(Q, action_holder)
# variable_summaries(R, '/R')

gamma_rew = tf.reshape(gamma * tf.reduce_max(next_Q, reduction_indices=1) * (1 - is_done_holder), (-1, 1, 1))
next_R = r_holder + gamma_rew
# variable_summaries(next_R, '/next_R')

# error with clipping
error = tf.clip_by_value(tf.reduce_mean((R - next_R)**2), clip_value_min=-1.0, clip_value_max=1.0)
variable_summaries(error, '/error')

# Define the operation that performs the optimization
learning_rate_holder = tf.placeholder(dtype=tf.float32, name='symbolic_state')
# training_step = tf.train.GradientDescentOptimizer(learning_rate_holder).minimize(error)
# training_step = tf.train.RMSPropOptimizer(learning_rate_holder).minimize(error)
# RMSProp parameters from BrainDQN
training_step = tf.train.RMSPropOptimizer(learning_rate_holder, 0.99, 0.0, 1e-6).minimize(error)

sess = tf.Session()  # FOR NOW everything is symbolic, this object has to be called to compute each value of Q
sess.run(tf.initialize_all_variables())

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
        return rd.choice(action_list)

    Q_values = sess.run(Q, feed_dict={state_holder: state.reshape(1, dim_state)})
    val = np.max(Q_values[0, :])
    max_indices = np.where(Q_values[0, 0, :] == val)[0]
    a = action_list[rd.choice(max_indices)]
    return a


time_list = []
reward_list = []
err_list = []
val_list = []
exp_mem_trials = []
steps = 0

for k in range(N_trial + N_trial_test):

    if k > N_trial:
        epsilon = 0
        learning_rate = 0

    acc_reward = 0  # Init the accumulated reward
    observation = new_observation = env.reset()  # Init the state
    pro_observation = prepro(new_observation, observation)
    if steps < exploration_steps:
        action = rd.choice(action_list)
    else:
        action = policy(pro_observation)  # Init the first action

    trial_err_list = []

    t = 0
    exp_mem = []
    # for t in range(trial_duration):
    while True:
        if render and k > 150: env.render()

        new_observation, reward, done, info = env.step(action)  # Take the action
        steps += 1
        pro_new_observation = prepro(new_observation, observation)

        exp_mem.append(zip((pro_observation, pro_new_observation, action_list.index(action), reward, done)))

        if k % 9 == 0 and done:
            print("learning rate: ", learning_rate, "epsilon:", epsilon)

        one_hot_action = np.zeros((1, n_action))
        one_hot_action[0, action_list.index(action)] = 1.0
        # Compute the Bellman Error for monitoring
        err = sess.run(error, feed_dict={
            state_holder: pro_observation.reshape(-1, dim_state),
            next_state_holder: pro_new_observation.reshape(-1, dim_state),
            action_holder: one_hot_action.reshape(-1, n_action, 1),
            is_done_holder: np.array(done).reshape(-1,),
            r_holder: np.array(reward).reshape(-1, 1, 1)})
        # Add the error in a trial-specific list of errors
        trial_err_list.append(err)

        observation = new_observation  # Pass the new state to the next step
        pro_observation = pro_new_observation  # Pass the new state to the next step
        if steps < exploration_steps:
            action = rd.choice(action_list)
        else:
            action = policy(pro_observation)  # Decide next action based on policy

        acc_reward += reward  # Accumulate the reward
        if reward != 0:
            if (reward < 0 and exp_neg <= exp_pos) or (reward > 0 and exp_pos <= exp_neg):
                exp_mem_trials.append(exp_mem)
                if reward > 0:
                    exp_pos += 1
                else:
                    exp_neg += 1
            exp_mem = []
        if done:
            break  # Stop the trial when the environment says it is done

        t += 1

    mb_size = 32
    exp_mem_trials = exp_mem_trials[-(exploration_period * 10):]

    # if len(exp_mem_trials) >= exploration_period and k % mb_size == 0:
    if len(exp_mem_trials) >= exploration_period:

        for i, batch in enumerate(iterate_minibatches(exp_mem_trials, shuffle=True, batchsize=1)):

            # if i >= mb_size:  # learn for i mini batches
            if i > 0:  # learn only once
                break

            minibatch_zip_ = batch[0]
            minibatch_zip = copy.deepcopy(minibatch_zip_)
            mb_obs, mb_nob, mb_act, mb_rew, mb_don = zip(*minibatch_zip)
            batch_len = len(mb_obs)

            mb_rew_sum = sum([r[0] for r in mb_rew])

            # reward propagation for previous frames
            if mb_rew_sum > 0:
                mb_rew = [(mb_rew_sum * ((i+1) / batch_len),) for i in range(batch_len)]
            if mb_rew_sum < 0:
                # mb_rew = [(mb_rew_sum * ((i+1) / batch_len)**4,) for i in range(batch_len)]
                mb_rew = [neg_prop(mb_rew_sum, i, batch_len) for i in range(batch_len)]

            mb_obs = np.array(mb_obs)  # [o  for o, no, a, r, d in minibatch_zip])
            mb_nob = np.array(mb_nob)  # [no for o, no, a, r, d in minibatch_zip])
            mb_act = np.array(mb_act)  # [a  for o, no, a, r, d in minibatch_zip])
            mb_rew = np.array(mb_rew)  # [r  for o, no, a, r, d in minibatch_zip])
            mb_don = np.array(mb_don).astype(np.float32)  # [d  for o, no, a, r, d in minibatch_zip])
            one_hot_actions = np.zeros((batch_len, n_action))
            one_hot_actions[np.arange(batch_len), np.reshape(mb_act, (batch_len,))[:]] = 1.0

            # Perform one step of gradient descent
            summary, _ = sess.run([merged, training_step], feed_dict={
                state_holder: mb_obs.reshape(-1, dim_state),
                next_state_holder: mb_nob.reshape(-1, dim_state),
                action_holder: one_hot_actions.reshape(-1, n_action, 1),
                is_done_holder: mb_don.reshape(-1,),
                r_holder: mb_rew.reshape(-1, 1, 1),
                learning_rate_holder: learning_rate})
            train_writer.add_summary(summary, steps)

        if learning_rate > 0.00025:
            learning_rate *= lr_decay
        else:
            learning_rate = 0.00025
        if epsilon > epsilon_min:
            epsilon *= epsi_decay
        else:
            epsilon = epsilon_min
    else:
        full = len(exp_mem_trials) / exploration_period
        print("%02d%% of memory ready" % int(full * 100))

    # Stack values for monitoring
    err_list.append(np.mean(trial_err_list))
    time_list.append(t + 1)
    reward_list.append(acc_reward)  # Store the result

    if steps < exploration_steps:
        print_results(k, time_list, err_list, reward_list, N_print_every=N_print_every)

plot_results(N_trial, N_trial_test, reward_list, time_list, err_list)
plt.show()
