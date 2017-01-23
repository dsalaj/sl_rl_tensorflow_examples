#!/usr/bin/env python3

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


# DQN parameters
network_folder_name = "saved_networks"
observe_steps = 50000
explore_steps = 1000000
min_epsilon = 0.1
init_epsilon = 1.
epsilon = init_epsilon
learning_rate = 0.00025
gamma = 0.99
replay_memory_size = 200000
mb_size = 32
update_target_net_every_frame_steps = 10000
batch_iter_num = 1
train_frequency = 1
repeat_action = 1
random_start_acts_max = 30

# General parameters
render = False
N_print_every = 10
N_trial = 1000000
N_trial_test = 100

# Generate the environment
env = gym.make("Pong-v0")
# dim_state = env.observation_space.high.__len__()
dim_state = 6  # after preprocessing it's 6
# n_action = env.action_space.n
n_action = 3
action_list = [0, 2, 3]  # only 3 controls used

img_size = 84
input_ch_num = 2
conv1_filter_size = 8
conv2_filter_size = 4
conv1_feature_maps_num = 16
conv2_feature_maps_num = 32
fc1_size = 256
out_size = n_action


# Used to set target network parameters to those of trained network
# Reference: https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


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
# Reference: https://www.tensorflow.org/tutorials/mnist/pros/
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Input N×N convolutional layer with filter size m×m, stride s
# output will be of size (N−m)/s + 1 × (N−m)/s + 1
def calc_convout_size(input_size, filter_size, stride):
    return (input_size - filter_size) / stride + 1


def get_out(input_, name, w_conv1, b_conv1, w_conv2, b_conv2, conv2_out_size, w_fc1, b_fc1, w_out, b_out):
    conv1 = tf.nn.relu(
        tf.nn.conv2d(input_, w_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1,
        name=name+"_conv1")
    conv2 = tf.nn.relu(
        tf.nn.conv2d(conv1, w_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2,
        name=name+"_conv2")

    conv2_flat = tf.reshape(conv2, [-1, conv2_out_size * conv2_out_size * conv2_feature_maps_num])
    fc1 = tf.nn.relu(tf.matmul(conv2_flat, w_fc1) + b_fc1, name=name+"_fc1")
    Q = tf.matmul(fc1, w_out) + b_out
    return Q


class Q_Network():
    def __init__(self, scope):
        with tf.variable_scope(scope):
            self.w_conv1 = weight_variable([conv1_filter_size, conv1_filter_size, input_ch_num, conv1_feature_maps_num])
            self.b_conv1 = bias_variable([conv1_feature_maps_num])
            self.w_conv2 = weight_variable([conv2_filter_size, conv2_filter_size, 16, conv2_feature_maps_num])
            self.b_conv2 = bias_variable([conv2_feature_maps_num])

            self.conv1_out_size = int(calc_convout_size(img_size, conv1_filter_size, 4))
            self.conv2_out_size = int(calc_convout_size(self.conv1_out_size, conv2_filter_size, 2))
            self.w_fc1 = weight_variable([self.conv2_out_size * self.conv2_out_size * conv2_feature_maps_num, 256])
            self.b_fc1 = bias_variable([256])
            self.w_out = weight_variable([256, out_size])
            self.b_out = bias_variable([out_size])

            self.state_holder = tf.placeholder(dtype=tf.float32, shape=(None, img_size, img_size, input_ch_num),
               name='symbolic_state1')
            self.target_Q = tf.placeholder(dtype=tf.float32, shape=(None, n_action), name='symbolic_state2')

            self.Q = get_out(self.state_holder, "Q", self.w_conv1, self.b_conv1, self.w_conv2, self.b_conv2,
                             self.conv2_out_size, self.w_fc1, self.b_fc1, self.w_out, self.b_out)
            self.Q = tf.reshape(self.Q, (-1, 1, n_action))

            if scope != 'main':
                self.update_target_network = update_target_graph('main', scope)
            else:
                self.action_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_action, 1), name='symbolic_action')
                self.r_holder = tf.placeholder(dtype=tf.float32, shape=(None, 1, 1), name='symbolic_value_estimation')
                self.is_done_holder = tf.placeholder(dtype=tf.float32, shape=(None,), name='is_done')

                R = tf.batch_matmul(self.Q, self.action_holder)

                gamma_rew = tf.reshape(
                    gamma * tf.reduce_max(self.target_Q, reduction_indices=1) * (1 - self.is_done_holder), (-1, 1, 1))
                next_R = self.r_holder + gamma_rew

                self.error = tf.reduce_mean(tf.square((R - next_R)))
                variable_summaries(self.error, '/error')

# main network which we train on every step
Q_network = Q_Network('main')
# target network which is used to make decisions in policy
target_Q_network = Q_Network('target')

# Define the operation that performs the optimization
training_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, momentum=0.0, epsilon=1e-6)\
    .minimize(Q_network.error)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(network_folder_name)
# if checkpoint and checkpoint.model_checkpoint_path:
#         saver.restore(sess, checkpoint.model_checkpoint_path)
#         print("Loaded weights from:", checkpoint.model_checkpoint_path)
# else:
#         print("Weights were initialised randomly.")

suffix = time.strftime('%Y-%m-%d--%H-%M-%S')
train_writer = tf.train.SummaryWriter('tensorboard/cartpole/{}'.format(suffix) + '/train', sess.graph)
merged = tf.merge_all_summaries()


# take random number (min 10) of random actions at start
def random_start():
    rand_acts = np.random.randint(10, random_start_acts_max)
    for act_counter in range(rand_acts):
        o, r, d, i = env.step(action_list[rd.choice(n_action)])
    return o


# epsilon greedy policy
def policy(state):
    if rd.rand() < epsilon:
        return rd.choice(n_action)

    Q_values = sess.run(target_Q_network.Q, feed_dict={
        target_Q_network.state_holder: state.reshape(1, img_size, img_size, input_ch_num)})
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
    if k > N_trial:
        epsilon = 0
        learning_rate = 0

    acc_reward = 0  # Init the accumulated reward
    obs1 = obs2 = obs3 = obs4 = env.reset()  # Init the state
    obs1 = obs2 = obs3 = obs4 = random_start()

    # for input we preprocess two frames with two steps in between for better motion information extraction
    pro_observation = prepro_14(obs1, obs4)
    action = policy(pro_observation)

    trial_err_list = []

    t = 0
    point_length = 0

    while True:
        if render: env.render()
        if frames_processed == 0:
            print("Start OBSERVATION...")
        if frames_processed == observe_steps:
            print("Start EXPLORATION...")
        if frames_processed == observe_steps + explore_steps:
            print("Start TRAINING...")

        reward = reward1 = 0
        for act_step in range(repeat_action):  # take action
            obs4, reward1, done, info = env.step(action_list[action])
            reward += reward1
            if done or reward != 0:
                break

        pro_new_observation = prepro_14(obs1, obs4)

        exp_mem.append(list(zip((pro_observation, pro_new_observation, action, reward, done))))
        frames_processed += 1
        point_length += 1

        # in observation period we do not train
        if frames_processed < observe_steps:
            obs1 = obs2
            obs2 = obs3
            obs3 = obs4
            pro_observation = pro_new_observation  # Pass the new state to the next step
            action = policy(pro_observation)
            acc_reward += reward
            t += 1
            if done:
                break
            continue

        # when the memory is full, pop the oldest entry
        if len(exp_mem) > replay_memory_size:
            exp_mem.pop(0)

        # update target network with parameters from the trained network
        if frames_processed % update_target_net_every_frame_steps == 0:
            sess.run(target_Q_network.update_target_network)
            # print("Exporting network to: ", network_folder_name + '/network' + '-parl')
            # if not os.path.exists(network_folder_name + '/'):
            #     os.makedirs(network_folder_name + '/')
            # saver.save(sess, network_folder_name + '/network' + '-parl', global_step=frames_processed)

        if frames_processed % train_frequency == 0:
            batch = random.sample(exp_mem, mb_size)

            minibatch_zip_ = batch
            minibatch_zip = copy.deepcopy(minibatch_zip_)
            mb_obs, mb_nob, mb_act, mb_rew, mb_don = zip(*minibatch_zip)

            mb_obs = np.array(mb_obs)
            mb_nob = np.array(mb_nob)
            mb_act = np.array(mb_act)
            mb_rew = np.array(mb_rew)
            mb_don = np.array(mb_don).astype(np.float32)
            one_hot_actions = np.zeros((mb_size, n_action))
            one_hot_actions[np.arange(mb_size), np.reshape(mb_act, (mb_size,))[:]] = 1.

            # Get the Q values from the target network
            target_Q = sess.run(target_Q_network.Q, feed_dict={
                target_Q_network.state_holder: mb_nob.reshape((-1, img_size,img_size,input_ch_num)),
            })
            # Perform one step of gradient descent
            summary, _ = sess.run([merged, training_step], feed_dict={
                Q_network.state_holder: mb_obs.reshape(-1, img_size,img_size,input_ch_num),
                Q_network.action_holder: one_hot_actions.reshape(-1, n_action, 1),
                Q_network.is_done_holder: mb_don.reshape(-1,),
                Q_network.r_holder: mb_rew.reshape(-1, 1, 1),
                Q_network.target_Q: target_Q.reshape(-1, n_action)
            })
            train_writer.add_summary(summary, summary_counter)
            summary_counter += 1
            descent_steps += 1

            # epsilon decay
            if epsilon > 0.1:
                epsilon -= (init_epsilon - min_epsilon) / explore_steps
            else:
                epsilon = 0.1

        one_hot_action = np.zeros((1, n_action))
        one_hot_action[0, action] = 1.
        # Compute the Bellman Error for monitoring
        target_Q = sess.run(target_Q_network.Q, feed_dict={
            target_Q_network.state_holder: pro_new_observation.reshape((-1, img_size,img_size,input_ch_num))
        })

        err = sess.run(Q_network.error, feed_dict={
            Q_network.state_holder: pro_observation.reshape((-1,img_size,img_size,input_ch_num)),
            Q_network.action_holder: one_hot_action.reshape(-1, n_action, 1),
            Q_network.is_done_holder: np.array(done).reshape(-1,),
            Q_network.r_holder: np.array(reward).reshape(-1, 1, 1),
            Q_network.target_Q: target_Q.reshape(-1, n_action)
        })
        trial_err_list.append(err)

        # Pass the new state to the next step
        obs1 = obs2
        obs2 = obs3
        obs3 = obs4
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

    if frames_processed > observe_steps and len(err_list) != 0:
        print_results(k, time_list, err_list, reward_list, N_print_every=N_print_every)

plot_results(N_trial, N_trial_test, reward_list, time_list, err_list)
plt.show()
