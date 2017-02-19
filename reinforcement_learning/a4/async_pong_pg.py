import threading
import multiprocessing
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from helper import *
import pong_tools

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

n_hidden = 50


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # Pong
            self.state_holder = tf.placeholder(dtype=tf.float32,
                                          shape=(None, n_obs),
                                          name='symbolic_state')
            self.actions_one_hot_holder = tf.placeholder(dtype=tf.float32,
                                                    shape=(None, n_actions),
                                                    name='symbolic_actions_one_hot_holder')
            self.discounted_rewards_holder = tf.placeholder(dtype=tf.float32,
                                                       shape=None,
                                                       name='symbolic_reward')

            theta_h = tf.get_variable('theta_h', shape=(n_obs, n_hidden),
                                      initializer=tf.random_normal_initializer(
                                          stddev=1.0 / np.sqrt(n_obs)))
            b_h = tf.get_variable('b_h', shape=n_hidden,
                                  initializer=tf.constant_initializer(0.0))

            theta = tf.get_variable('theta', shape=(n_hidden, n_actions),
                                    initializer=tf.random_normal_initializer(
                                        stddev=1.0 / np.sqrt(n_obs)))
            b = tf.get_variable('b', shape=n_actions,
                                initializer=tf.constant_initializer(0.0))

            # action_probabilies: dim = traj-length x n_actions, softmax over the output. Used for action selection in the
            # training loop
            a_y = tf.matmul(self.state_holder, theta_h, name='output_activation') + b_h
            y = tf.nn.tanh(a_y)
            a_z = tf.matmul(y, theta, name='output_activation') + b
            self.action_probabilities = tf.nn.softmax(a_z, name='action_probabilities')

            # This operation is used for action selection during testing, to select the action with the maximum action probability
            testing_action_choice = tf.argmax(self.action_probabilities, dimension=1,
                                              name='testing_action_choice')
            chosen_action_prob = tf.reduce_sum(
                self.action_probabilities * tf.reshape(self.actions_one_hot_holder,
                                                  (-1, n_actions)), 1)
            L_theta = - tf.reduce_sum(
                tf.log(chosen_action_prob)) * tf.reduce_sum(
                self.discounted_rewards_holder)
            self.loss = L_theta

            # grad_apply = tf.train.RMSPropOptimizer(learning_rate).minimize(L_theta)
            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                # self.var_norms = tf.global_norm(local_vars)
                # grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
                grads = self.gradients

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
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        observations = [row.tolist() for row in observations]
        actions = rollout[:, 1]
        actions = [row.tolist() for row in actions]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # rnn_state = self.local_AC.state_init
        feed_dict = {
                     self.local_AC.discounted_rewards_holder: rewards,
                     self.local_AC.state_holder: observations,
                     self.local_AC.actions_one_hot_holder: actions,
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

                s = s1 = self.env.reset()
                episode_frames.append(s)
                s = pong_tools.prepro(s1, s)
                # rnn_state = self.local_AC.state_init
                done = False

                while not done:
                    # Take an action using probabilities from policy network output.
                    action_probability_values = sess.run(
                        self.local_AC.action_probabilities,
                        feed_dict={self.local_AC.state_holder: s.reshape(1, n_obs)})
                    action_idx = np.random.choice(range(n_actions), p=action_probability_values.ravel())
                    a = action_list[action_idx]
                    # Calculating the one-hot action array for use by tensorflow
                    action_arr = np.zeros(n_actions)
                    action_arr[action_idx] = 1.

                    s1, r, done, info = self.env.step(a)
                    episode_frames.append(s1)
                    if not done:
                        s1 = pong_tools.prepro(episode_frames[-1], episode_frames[-2])
                    else:
                        s1 = s

                    episode_buffer.append([s, action_arr, r, s1, done, None])
                    # Update the network using the experience buffer at the end of the episode.
                    if r != 0 or (len(episode_buffer) > 0 and done):
                        loss = self.train(global_AC, episode_buffer, sess, gamma, 0.0)
                        sess.run(self.update_local_ops)
                        episode_buffer = []

                    episode_reward += r
                    s = s1
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

