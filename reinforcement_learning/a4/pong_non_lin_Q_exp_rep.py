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

# Algorithm parameters
# learning_rate = .01
# gamma = .9
# epsilon = .7
# epsi_decay = .99999
# lr_decay = .99999
# n_hidden = 60


#DQN Paper parameters
observe_steps = 50000
explore_steps = 1000000
min_epsilon = 0.1
init_epsilon = 1.
epsilon = 1.
learning_rate = 0.00025
lr_decay = 0.99
gamma = 0.95
replay_memory_size = 1000000
mb_size = 32

n_hidden = 32

# General parameters
render = False
# render = True
N_print_every = 10
N_trial = 1000000
N_trial_test = 100
trial_duration = 200

# Generate the environment
env = gym.make("Pong-v0")
# dim_state = env.observation_space.high.__len__()
dim_state = 6  # after preprocessing it's 6
# n_action = env.action_space.n
n_action = 3
action_list = [0, 2, 3]  # only 3 controls used


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

# wh0 = np.float32(rd.normal(loc=0.0, scale=1. / np.sqrt(dim_state), size=(dim_state, n_hidden)))
# bh0 = np.zeros(n_hidden, dtype=np.float32)
# w0 = np.float32(rd.normal(loc=0.0, scale=1. / np.sqrt(n_hidden), size=(n_hidden, n_action)))
# b0 = np.zeros(n_action, dtype=np.float32)
#
# # Generate the symbolic variables to hold the state values
# state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')
# next_state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')
#
# # Create the parameters of the Q model
# w = tf.Variable(initial_value=w0, trainable=True, name='weight_variable')
# b = tf.Variable(tf.constant(0.1, shape=[n_action]), trainable=True, name='bias')
# wh = tf.Variable(initial_value=wh0, trainable=True, name='hidden_weight_variable')
# bh = tf.Variable(tf.constant(0.1, shape=[n_hidden]), trainable=True, name='hidden_bias')
#
# # Q function at the current step
# a_y = tf.matmul(state_holder, wh, name='output_activation') + bh
# y = tf.nn.relu(a_y, name='hidden_layer_activation')
# a_z = tf.matmul(y, w, name='output_activation') + b
# Q = tf.nn.tanh(a_z, name='Q_model')
# # Q = a_z
#
# # Q function at the next step
# next_a_y = tf.matmul(next_state_holder, wh, name='next_step_output_activation') + bh
# next_y = tf.nn.relu(next_a_y, name='next_step_hidden_layer_activation')
# next_a_z = tf.matmul(next_y, w, name='next_step_output_activation') + b
# next_Q = tf.nn.tanh(next_a_z, name='next_step_Q_model')
# # next_Q = next_a_z


# Initialize the parameters of the Q model
wh1 = np.float32(rd.normal(loc=0.0, scale=1./np.sqrt(dim_state), size=(n_hidden, n_hidden)))
bh1 = np.zeros(n_hidden, dtype=np.float32)
wh0 = np.float32(rd.normal(loc=0.0, scale=1./np.sqrt(dim_state), size=(dim_state, n_hidden)))
bh0 = np.zeros(n_hidden, dtype=np.float32)
w0 = np.float32(rd.normal(loc=0.0, scale=1./np.sqrt(n_hidden), size=(n_hidden, n_action)))
b0 = np.zeros(n_action, dtype=np.float32)

# Generate the symbolic variables to hold the state values
state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')
next_state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')

# Create the parameters of the Q model
w = tf.Variable(initial_value=w0, trainable=True, name='weight_variable')
b = tf.Variable(tf.constant(0.1, shape=[n_action]), trainable=True, name='bias')
wh = tf.Variable(initial_value=wh0, trainable=True, name='hidden_weight_variable')
bh = tf.Variable(tf.constant(0.1, shape=[n_hidden]), trainable=True, name='hidden_bias')
wh1 = tf.Variable(initial_value=wh1, trainable=True, name='hidden_weight_variable1')
bh1 = tf.Variable(tf.constant(0.1, shape=[n_hidden]), trainable=True, name='hidden_bias1')



# Q function at the current step
a_y = tf.matmul(state_holder, wh, name='output_activation') + bh
y = tf.nn.relu(a_y, name='hidden_layer_activation')
a_y1 = tf.matmul(y, wh1, name='output_activation') + bh1
y1 = tf.nn.relu(a_y1, name='hidden_layer_activation')
a_z1 = tf.matmul(y1, w, name='output_activation') + b

# Q = tf.nn.tanh(a_z1, name='Q_model')
Q = a_z1

# Q function at the next step
next_a_y = tf.matmul(next_state_holder, wh, name='next_step_output_activation') + bh
next_y = tf.nn.relu(next_a_y, name='next_step_hidden_layer_activation')
next_a_y1 = tf.matmul(next_y, wh1, name='next_step_output_activation') + bh1
next_y1 = tf.nn.relu(next_a_y1, name='next_step_hidden_layer_activation')
next_a_z1 = tf.matmul(next_y1, w, name='next_step_output_activation') + b

# next_Q = tf.nn.tanh(next_a_z1, name='next_step_Q_model')
next_Q = next_a_z1

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

error = tf.reduce_mean((R - next_R)**2)
variable_summaries(error, '/error')

# Define the operation that performs the optimization
learning_rate_holder = tf.placeholder(dtype=tf.float32, name='symbolic_state')
training_step = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.99, momentum=0.0, epsilon=1e-6).minimize(error)
# training_step = tf.train.GradientDescentOptimizer(learning_rate_holder).minimize(error)

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
exp_mem = []
skip_counter = 0
frames_processed = 0

for k in range(N_trial + N_trial_test):
    if k % 10000 == 0:
        print("Current iter k: ", k)
    if k > N_trial:
        epsilon = 0
        learning_rate = 0

    acc_reward = 0  # Init the accumulated reward
    observation = new_observation = env.reset()  # Init the state
    pro_observation = prepro(observation, new_observation)
    action = policy(pro_observation)  # Init the first action

    trial_err_list = []

    t = 0
    # for t in range(trial_duration):
    while True:
        if render: env.render()
        if (frames_processed == 0):
            print("Start OBSERVATION...")
        if (frames_processed == observe_steps):
            print("Start EXPLORATION...")
        if (frames_processed == observe_steps + explore_steps):
            print("Start TRAINING...")

        new_observation, reward, done, info = env.step(action)  # Take the action
        # rms
        # reward *= 10
        pro_new_observation = prepro(observation, new_observation)

        # print(pro_observation)
        # print(pro_new_observation)
        # print("-------------")

        # if(reward != 0):
        #
        exp_mem.append(list(zip((pro_observation, pro_new_observation, action_list.index(action), reward, done))))
        frames_processed += 1


        if frames_processed < observe_steps:
            observation = new_observation  # Pass the new state to the next step
            pro_observation = pro_new_observation  # Pass the new state to the next step
            action = policy(pro_observation)
            if done:
                break  # Stop the trial when the environment says it is done
            continue

        if (len(exp_mem) > replay_memory_size):
            exp_mem.pop(0)
            # exp_mem = sorted(exp_mem , key=lambda tup: tup[3])
            # exp_mem = exp_mem[:40] + exp_mem[60:]
            # print("New Memory size: ", len(exp_mem))
            # print("First score: ", exp_mem[0][3])
            # print("Mid score: ", exp_mem[int(len(exp_mem)/2)][3])
            # print("Last score: ", exp_mem[-1][3])
            # a = 0
            # b = 0
            # for mem_i in range(len(exp_mem)):
            #     if(exp_mem[mem_i][3][0] == -1): a +=1
            #     elif(exp_mem[mem_i][3][0] == 1): b +=1
            # print("Number of -1s: ", a)
            # print("Number of 1s: ", b)
        if (k % 10 == 0 and done):
            # print("Current learning rate: ", learning_rate)
            print("Current epsilon: ", epsilon)
            print("Memory Length: ", len(exp_mem))
            if(len(err_list) > 0):
                print("Last error: ", err_list[-1])

        # if len(exp_mem) >= mb_size and t % mb_size == 0:
        # if done or reward != 0:
        # if len(exp_mem) >= mb_size * 2:
        if True:
            for i, batch in enumerate(iterate_minibatches(exp_mem, shuffle=True, batchsize=mb_size)):

                # print("--------------------------------")
                # print(batch)
                # print("--------------------------------")
                # if i > 5:  # learn for 5 mini batches
                #     break

                minibatch_zip_ = batch
                minibatch_zip = copy.deepcopy(minibatch_zip_)
                mb_obs, mb_nob, mb_act, mb_rew, mb_don = zip(*minibatch_zip)

                mb_obs = np.array(mb_obs)  # [o  for o, no, a, r, d in minibatch_zip])
                mb_nob = np.array(mb_nob)  # [no for o, no, a, r, d in minibatch_zip])
                mb_act = np.array(mb_act)  # [a  for o, no, a, r, d in minibatch_zip])
                mb_rew = np.array(mb_rew)  # [r  for o, no, a, r, d in minibatch_zip])
                mb_don = np.array(mb_don).astype(np.float32)  # [d  for o, no, a, r, d in minibatch_zip])
                one_hot_actions = np.zeros((mb_size, n_action))
                one_hot_actions[np.arange(mb_size), np.reshape(mb_act, (mb_size,))[:]] = 1

                # Perform one step of gradient descent
                summary, _ = sess.run([merged, training_step], feed_dict={
                    state_holder: mb_obs.reshape(-1, dim_state),
                    next_state_holder: mb_nob.reshape(-1, dim_state),
                    action_holder: one_hot_actions.reshape(-1, n_action, 1),
                    is_done_holder: mb_don.reshape(-1,),
                    r_holder: mb_rew.reshape(-1, 1, 1),
                    learning_rate_holder: learning_rate})
                train_writer.add_summary(summary, k*trial_duration + t)

                #TODO maybe train one batch per iteration (break), maybe more
                # if i > 5:
                #     break
                break

            # if learning_rate > 1e-4:
            #     learning_rate *= lr_decay
            # else:
            #     learning_rate = 1e-4
            # learning_rate *= lr_decay
            if epsilon > 0.1 and frames_processed < explore_steps + observe_steps:
                epsilon -= (init_epsilon - min_epsilon) / explore_steps
            else:
                epsilon = 0.1

            # exp_mem = []

        one_hot_action = np.zeros((1, n_action))
        one_hot_action[0, action_list.index(action)] = 1
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
        action = policy(pro_observation)  # Decide next action based on policy
        acc_reward += reward  # Accumulate the reward

        if done:
            break  # Stop the trial when the environment says it is done

        t += 1

    # Stack values for monitoring
    err_list.append(np.mean(trial_err_list))
    time_list.append(t + 1)
    reward_list.append(acc_reward)  # Store the result

    if(frames_processed > explore_steps):
        print_results(k, time_list, err_list, reward_list, N_print_every=N_print_every)

plot_results(N_trial, N_trial_test, reward_list, time_list, err_list)
plt.show()
