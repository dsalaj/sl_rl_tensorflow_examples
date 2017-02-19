#!/usr/bin/env python

import os
import gym
import time
import copy
import random
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from image_tools import prepro_14
from cartpole_utils import plot_results,print_results
from tf_tools import variable_summaries
import tensorflow as tf


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# DQN Paper parameters
observe_steps = 50000
explore_steps = 50000
min_epsilon = 0.1
init_epsilon = 1.
epsilon = 1.
learning_rate = 0.00025
lr_decay = 0.99
gamma = 0.95
replay_memory_size = 50000
mb_size = 32

# General parameters
render = False
# render = True
N_print_every = 10
N_trial = 1000000
N_trial_test = 100
# trial_duration = 200

# Generate the environment
env = gym.make("Pong-v0")
# dim_state = env.observation_space.high.__len__()
dim_state = 6  # after preprocessing it's 6
n_action = env.action_space.n
# action list should be same for all games
# n_action = 3
# action_list = [0, 2, 3]  # only 3 controls used
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

# Functions for initializing parameters of the network
#Reference: https://www.tensorflow.org/tutorials/mnist/pros/
#----------------------------------------------------------
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Input N×N
# convolutional layer with filter size m×m, stride s
# output will be of size (N−m)/s + 1 × (N−m)/s + 1
def calc_convout_size(input_size, filter_size, stride):
    return (input_size - filter_size) / stride + 1


def get_out(input_, name, w_conv1, b_conv1, w_conv2, b_conv2, conv2_out_size, w_fc1, b_fc1, w_out, b_out):
    conv1 = tf.nn.relu(tf.nn.conv2d(input_, w_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name=name+"_conv1")
    # conv1 = tf.nn.batch_normalization(conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name=name+"_conv2")
    # conv2 = tf.nn.batch_normalization(conv2)

    conv2_flat = tf.reshape(conv2, [-1, conv2_out_size * conv2_out_size * conv2_feature_maps_num])
    fc1 = tf.nn.relu(tf.matmul(conv2_flat, w_fc1) + b_fc1, name=name+"_fc1")
    Q = tf.matmul(fc1, w_out) + b_out
    return Q


class Q_Network():
    def __init__(self, scope):
        with tf.variable_scope(scope):
            # Initialize the parameters of the Q model
            self.w_conv1 = weight_variable([conv1_filter_size, conv1_filter_size, input_ch_num, conv1_feature_maps_num])
            self.b_conv1 = bias_variable([conv1_feature_maps_num])
            self.w_conv2 = weight_variable([conv2_filter_size, conv2_filter_size, 16, conv2_feature_maps_num])
            self.b_conv2 = bias_variable([conv2_feature_maps_num])

            self.conv1_out_size = int(calc_convout_size(img_size, conv1_filter_size, 4))
            self.conv2_out_size = int(calc_convout_size(self.conv1_out_size, conv2_filter_size, 2))
            print("conv out size: ", self.conv2_out_size)
            self.w_fc1 = weight_variable([self.conv2_out_size * self.conv2_out_size * conv2_feature_maps_num, 256])
            self.b_fc1 = bias_variable([256])
            self.w_out = weight_variable([256, out_size])
            self.b_out = bias_variable([out_size])

            # Generate the symbolic variables to hold the state values
            self.state_holder = tf.placeholder(dtype=tf.float32, shape=(None, img_size, img_size, input_ch_num), name='symbolic_state')
            self.next_state_holder = tf.placeholder(dtype=tf.float32, shape=(None, img_size, img_size, input_ch_num), name='symbolic_state')

            Q = get_out(self.state_holder, "Q", self.w_conv1, self.b_conv1, self.w_conv2, self.b_conv2, self.conv2_out_size, self.w_fc1, self.b_fc1, self.w_out, self.b_out)
            self.next_Q = get_out(self.next_state_holder, "next_Q", self.w_conv1, self.b_conv1, self.w_conv2, self.b_conv2, self.conv2_out_size, self.w_fc1, self.b_fc1, self.w_out, self.b_out)

            self.Q = tf.reshape(Q, (-1, 1, n_action))
            if scope != 'main':
                self.update_target_network = update_target_graph('main', scope)

# Define symbolic variables that will carry information needed for training
action_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_action, 1), name='symbolic_action')
r_holder = tf.placeholder(dtype=tf.float32, shape=(None, 1, 1), name='symbolic_value_estimation')
is_done_holder = tf.placeholder(dtype=tf.float32, shape=(None,), name='is_done')

Q_network = Q_Network('main')
target_Q_network = Q_Network('target')

# define the role of each training step
R = tf.batch_matmul(Q_network.Q, action_holder)
# variable_summaries(R, '/R')

gamma_rew = tf.reshape(gamma * tf.reduce_max(Q_network.next_Q, reduction_indices=1) * (1 - is_done_holder), (-1, 1, 1))
# gamma_rew = tf.reshape(gamma * tf.reduce_max(next_Q, reduction_indices=1)
#  * (1 - tf.abs(tf.reshape(r_holder,(-1,)))), (-1, 1, 1))
next_R = r_holder + gamma_rew
# variable_summaries(next_R, '/next_R')

error = tf.clip_by_value(tf.reduce_mean(tf.square((R - next_R))), clip_value_min=-1., clip_value_max=1.)
variable_summaries(error, '/error')

# Define the operation that performs the optimization
learning_rate_holder = tf.placeholder(dtype=tf.float32, name='symbolic_state')
training_step = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.99, momentum=0.0, epsilon=1e-6).minimize(error)
# training_step = tf.train.GradientDescentOptimizer(learning_rate_holder).minimize(error)

sess = tf.Session()  # FOR NOW everything is symbolic, this object has to be called to compute each value of Q
sess.run(tf.initialize_all_variables())

# saver = tf.train.Saver()
# checkpoint = tf.train.get_checkpoint_state("saved_networks")
# if checkpoint and checkpoint.model_checkpoint_path:
#         saver.restore(sess, checkpoint.model_checkpoint_path)
#         print("Loaded weights from:", checkpoint.model_checkpoint_path)
# else:
#         print("Weights were initialised randomly.")

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

    Q_values = sess.run(target_Q_network.Q, feed_dict={target_Q_network.state_holder: state.reshape(1, img_size, img_size, input_ch_num)})
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
    pro_observation = prepro_14(obs1, obs4)
    action = policy(pro_observation)  # Init the first action

    trial_err_list = []

    t = 0
    point_length = 0

    while True:
        if render: env.render()
        if (frames_processed == 0):
            print("Start OBSERVATION...")
        if (frames_processed == observe_steps):
            print("Start EXPLORATION...")
        if (frames_processed == observe_steps + explore_steps):
            print("Start TRAINING...")

        obs4, reward, done, info = env.step(action)  # Take the action
        # print("rew done", reward, done)
        # plt.imshow(obs4)
        # plt.show()
        # rms
        # reward *= 10

        # pro_new_observation = prepro(obs1, obs2, obs3, obs4)
        pro_new_observation = prepro_14(obs1, obs4)

        # plt.imshow(pro_new_observation[:,:,1], cmap='gray')
        # plt.show()
        # print("max pro: ", np.min(pro_new_observation[:,:,1]))
        # For observing frames
        # plt.figure(10);
        # plt.clf()
        # plt.imshow(pro_new_observation[:,:,1], cmap='gray');
        # plt.title('Camera Frame')
        # plt.pause(0.3)

        exp_mem.append(list(zip((pro_observation, pro_new_observation, action, reward, done))))
        frames_processed += 1
        point_length += 1

        if reward != 0:
            for i in range(point_length):
                exp_mem[-(i+1)][3] = reward * (gamma ** i)
            # exp_mem[-15:][3] = reward
            point_length = 0

        if frames_processed < observe_steps:
            obs1 = obs2
            obs2 = obs3
            obs3 = obs4
            pro_observation = pro_new_observation  # Pass the new state to the next step
            action = policy(pro_observation)
            acc_reward += reward
            t += 1
            if done:
                # print("done")
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
            print("Frames processed: ", frames_processed)
            if(len(err_list) > 0):
                print("Last error: ", err_list[-1])

        # if len(exp_mem) >= mb_size and t % mb_size == 0:
        # if done or reward != 0:
        # if len(exp_mem) >= mb_size * 2:
        if reward != 0:
            # for i, batch in enumerate(iterate_minibatches(exp_mem, shuffle=True, batchsize=mb_size)):
            batch = random.sample(exp_mem, mb_size)
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
            one_hot_actions[np.arange(mb_size), np.reshape(mb_act, (mb_size,))[:]] = 1.
            # Perform one step of gradient descent

            summary, _ = sess.run([merged, training_step], feed_dict={
                Q_network.state_holder: mb_obs.reshape(-1, img_size,img_size,input_ch_num),
                Q_network.next_state_holder: mb_nob.reshape(-1, img_size,img_size,input_ch_num),
                action_holder: one_hot_actions.reshape(-1, n_action, 1),
                is_done_holder: mb_don.reshape(-1,),
                r_holder: mb_rew.reshape(-1, 1, 1),
                learning_rate_holder: learning_rate})
            train_writer.add_summary(summary, summary_counter)
            summary_counter += 1
            descent_steps += 1
            if descent_steps % 300 == 0:
                sess.run(target_Q_network.update_target_network)

            #TODO maybe train one batch per iteration (break), maybe more
            # Change to iterate_minibatches for more minibatches
            # if i > 5:
            #     break
            #break

            # if learning_rate > 1e-4:
            #     learning_rate *= lr_decay
            # else:
            #     learning_rate = 1e-4
            # learning_rate *= lr_decay
            if epsilon > 0.1:
                epsilon -= (init_epsilon - min_epsilon) / explore_steps
            else:
                epsilon = 0.1

            # exp_mem = []

        one_hot_action = np.zeros((1, n_action))
        one_hot_action[0, action] = 1.
        # Compute the Bellman Error for monitoring
        err = sess.run(error, feed_dict={
            Q_network.state_holder: pro_observation.reshape((-1,img_size,img_size,input_ch_num)),
            Q_network.next_state_holder: pro_new_observation.reshape((-1,img_size,img_size,input_ch_num)),
            action_holder: one_hot_action.reshape(-1, n_action, 1),
            is_done_holder: np.array(done).reshape(-1,),
            r_holder: np.array(reward).reshape(-1, 1, 1)})
        # Add the error in a trial-specific list of errors
        trial_err_list.append(err)

        # Pass the new state to the next step
        obs1 = obs2
        obs2 = obs3
        obs3 = obs4
        pro_observation = pro_new_observation  # Pass the new state to the next step
        action = policy(pro_observation)  # Decide next action based on policy
        acc_reward += reward  # Accumulate the reward

        # if frames_processed % 100000 == 0 and frames_processed > observe_steps:
        #     #PARL - Playing Atari with reinforcement learning
        #     print("Exporting network to: ", 'saved_networks/' + 'network' + '-parl')
        #     if not os.path.exists('saved_networks/'):
        #         os.makedirs('saved_networks/')
        #     saver.save(sess, 'saved_networks/' + 'network' + '-parl', global_step=frames_processed)

        if done:
            break  # Stop the trial when the environment says it is done

        t += 1

    # Stack values for monitoring
    err_list.append(np.mean(trial_err_list))
    time_list.append(t + 1)
    reward_list.append(acc_reward)  # Store the result

    # save network every 100000 iteration

    if(frames_processed > observe_steps and len(err_list) != 0):
        print_results(k, time_list, err_list, reward_list, N_print_every=N_print_every)

plot_results(N_trial, N_trial_test, reward_list, time_list, err_list)
plt.show()
