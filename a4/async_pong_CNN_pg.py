import threading
import multiprocessing
import copy
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from helper import *
import pong_tools
from image_tools import prepro_14

from random import choice
from time import sleep
from time import time
import gym

n_actions = 3
action_list = [0, 2, 3]
# The *relative* y coordinate of the opponent and the x,y coordinates of the ball for *two* frames
n_obs = 6

print_per_episode = 20

n_train_trials = 1000
n_test_trials = 100
gamma = 0.9999
learning_rate = 0.001
random_start_acts_max = 30

img_size = 84
input_ch_num = 2
conv1_filter_size = 8
conv2_filter_size = 4
conv1_feature_maps_num = 16
conv2_feature_maps_num = 32
fc1_size = 256
out_size = n_actions


def random_start(env):
    rand_acts = np.random.randint(10, random_start_acts_max)
    for act_counter in range(rand_acts):
        o, r, d, i = env.step(action_list[rd.choice(n_actions)])
    return o


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


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # Pong
            # Generate the symbolic variables to hold the state values
            self.state_holder = tf.placeholder(dtype=tf.float32, shape=(None, img_size, img_size, input_ch_num),
                                               name='symbolic_state1')

            # Create the parameters of the Q model
            # Initialize the parameters of the Q model
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

            self.Q = get_out(self.state_holder, "Q", self.w_conv1, self.b_conv1, self.w_conv2, self.b_conv2, self.conv2_out_size, self.w_fc1, self.b_fc1, self.w_out, self.b_out)
            self.Q = tf.reshape(self.Q, (-1, 1, n_actions))
            self.action_probabilities = tf.nn.softmax(self.Q, name='action_probabilities')

            # Define symbolic variables that will carry information needed for training
            self.action_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_actions, 1), name='symbolic_action')
            self.r_holder = tf.placeholder(dtype=tf.float32, shape=(None, 1, 1), name='symbolic_value_estimation')

            # This operation is used for action selection during testing, to select the action with the maximum action probability
            # testing_action_choice = tf.argmax(self.action_probabilities, dimension=1,
            #                                   name='testing_action_choice')
            chosen_action_prob = tf.reduce_sum(
                self.action_probabilities *
                tf.reshape(self.action_holder, (-1, n_actions)), 1)
            L_theta = - tf.reduce_sum(tf.log(chosen_action_prob)) * tf.reduce_sum(self.r_holder)
            self.loss = L_theta

            # grad_apply = tf.train.RMSPropOptimizer(learning_rate).minimize(L_theta)
            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                # self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 1.0)
                # grads = self.gradients

                # Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars))


class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        # self.episode_mean_values = []
        # self.summary_writer = tf.train.SummaryWriter("train_" + str(self.number))
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.env = game

    def train(self, global_AC, rollout, sess, gamma, bootstrap_value):
        rollout_copy = copy.deepcopy(rollout)
        np_rollout = np.array(rollout_copy)
        observations = np_rollout[:, 0]
        observations = np.array([row.tolist() for row in observations])
        actions = np_rollout[:, 1]
        actions = np.array([row.tolist() for row in actions])
        rewards = np_rollout[:, 2]
        next_observations = np_rollout[:, 3]
        dones = np_rollout[:, 4]

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # rnn_state = self.local_AC.state_init
        feed_dict = {
                     self.local_AC.r_holder: rewards.reshape(-1, 1, 1),
                     self.local_AC.state_holder: observations.reshape(-1, img_size, img_size, input_ch_num),
                     self.local_AC.action_holder: actions.reshape(-1, n_actions, 1),
                     }
        loss, _ = sess.run([self.local_AC.loss, self.local_AC.apply_grads], feed_dict=feed_dict)
        return loss / len(rollout)

    def work(self, max_episode_length, gamma, global_AC, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                # episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0

                obs1 = obs2 = obs3 = obs4 = self.env.reset()  # Init the state
                # obs1 = obs2 = obs3 = obs4 = random_start(self.env)
                pro_observation = prepro_14(obs1, obs4)
                done = False

                while not done:
                    # Take an action using probabilities from policy network output.
                    action_probability_values = sess.run(
                        self.local_AC.action_probabilities,
                        feed_dict={self.local_AC.state_holder: pro_observation.reshape(-1, img_size, img_size, input_ch_num)})

                    action_idx = np.random.choice(range(n_actions), p=action_probability_values.ravel())
                    a = action_list[action_idx]
                    # Calculating the one-hot action array for use by tensorflow
                    action_arr = np.zeros(n_actions)
                    action_arr[action_idx] = 1.

                    obs4, reward, done, info = self.env.step(a)
                    pro_new_observation = prepro_14(obs1, obs4)

                    episode_buffer.append([pro_observation, action_arr, reward, pro_new_observation, done, None])
                    # Update the network using the experience buffer at the end of the episode.
                    if reward != 0 or (len(episode_buffer) > 0 and done):
                        loss = self.train(global_AC, episode_buffer, sess, gamma, 0.0)
                        sess.run(self.update_local_ops)
                        episode_buffer = []

                    episode_reward += reward
                    obs1 = obs2
                    obs2 = obs3
                    obs3 = obs4
                    pro_observation = pro_new_observation
                    total_steps += 1
                    episode_step_count += 1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

                if episode_count % print_per_episode == 0:
                    print("{}: in last {} episodes before episode {} avg REWARDS"
                          .format(self.name, print_per_episode, episode_count),
                          np.mean(self.episode_rewards[(episode_count - print_per_episode):episode_count]), '+-',
                          np.std(self.episode_rewards[(episode_count - print_per_episode):episode_count]),
                          "avg STEPS",
                          np.mean(self.episode_lengths[(episode_count - print_per_episode):episode_count]), '+-',
                          np.std(self.episode_lengths[(episode_count - print_per_episode):episode_count]),
                          )

max_episode_length = s_size = a_size = 0
tf.reset_default_graph()


with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes',
                                  trainable=False)
    # trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    trainer = tf.train.RMSPropOptimizer(learning_rate)
    master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network
    num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(
            Worker(gym.make("Pong-v0"), i, s_size, a_size, trainer, global_episodes)
        )
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma,
                                          master_network, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

