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

print_per_episode = 10

n_train_trials = 20000
n_test_trials = 100
gamma = 0.99
learning_rate = 0.00025

n_hidden = 60

target_update_steps = 1500
async_update_steps = 3000


class QNetwork():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):

            # Generate the symbolic variables to hold the state values
            self.state_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_obs), name='symbolic_state')
            self.next_state_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_obs), name='symbolic_state')

            # Create the parameters of the Q model
            w = tf.Variable(initial_value=tf.truncated_normal([n_hidden, n_actions], stddev=0.1), trainable=True, name='weight_variable')
            b = tf.Variable(initial_value=tf.constant(0.1, shape=[n_actions]), trainable=True, name='bias')
            wh = tf.Variable(initial_value=tf.truncated_normal([n_obs, n_hidden], stddev=0.1), trainable=True, name='hidden_weight_variable')
            bh = tf.Variable(initial_value=tf.constant(0.1, shape=[n_hidden]), trainable=True, name='hidden_bias')

            # Q function at the current step
            a_y = tf.matmul(self.state_holder, wh, name='output_activation') + bh
            y = tf.nn.relu(a_y, name='hidden_layer_activation')
            a_z = tf.matmul(y, w, name='output_activation') + b

            # # Q function at the next step
            # next_a_y = tf.matmul(self.next_state_holder, wh, name='next_step_output_activation') + bh
            # next_y = tf.nn.relu(next_a_y, name='next_step_hidden_layer_activation')
            # next_a_z = tf.matmul(next_y, w, name='next_step_output_activation') + b
            # self.next_Q = next_a_z
            self.next_Q = tf.placeholder(dtype=tf.float32, shape=(None, n_actions))
            self.Q = tf.reshape(a_z, (-1, 1, n_actions))

            # Define symbolic variables that will carry information needed for training
            self.action_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_actions, 1), name='symbolic_action')
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

            self.loss = tf.reduce_mean(tf.square(R - next_R))
            # variable_summaries(self.error, '/error')

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
                # print("local vars =", len(local_vars))
                # self.clear_grads = tf.assign(self.gradients, tf.constant(0.))


class TargetQNetwork():
    def __init__(self, s_size, a_size, scope, trainer, trained_scope):
        with tf.variable_scope(scope):

            # Generate the symbolic variables to hold the state values
            self.state_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_obs))
            self.next_state_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_obs))

            # Create the parameters of the Q model
            w = tf.Variable(initial_value=tf.truncated_normal([n_hidden, n_actions], stddev=0.1), trainable=True)
            b = tf.Variable(initial_value=tf.constant(0.1, shape=[n_actions]), trainable=True)
            wh = tf.Variable(initial_value=tf.truncated_normal([n_obs, n_hidden], stddev=0.1), trainable=True)
            bh = tf.Variable(initial_value=tf.constant(0.1, shape=[n_hidden]), trainable=True)

            # Q function at the current step
            a_y = tf.matmul(self.state_holder, wh) + bh
            y = tf.nn.relu(a_y)
            self.a_z = tf.matmul(y, w) + b

            self.Q = tf.reshape(self.a_z, (-1, 1, n_actions))

            self.update_target_network = update_target_graph(trained_scope, scope)


def policy(state, epsilon, target_Q_net):
    if rd.rand() < epsilon:
        return rd.choice(n_actions)

    Q_values = sess.run(target_Q_net.Q,
                        feed_dict={target_Q_net.state_holder: state.reshape(1, n_obs)})
    val = np.max(Q_values[0, :])
    max_indices = np.where(Q_values[0, 0, :] == val)[0]
    a = rd.choice(max_indices)
    return a


class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_errs = []
        # self.episode_mean_values = []
        self.summary_writer = tf.train.SummaryWriter("train_" + str(self.number))
        # self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.epsilon = 0.7

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_Q = QNetwork(s_size, a_size, self.name, trainer)
        self.local_target_Q = TargetQNetwork(s_size, a_size, self.name + "target", trainer, self.name)
        self.update_local_ops = update_target_graph('global', self.name)

        self.env = game

    def train(self, global_Q, rollout, sess, gamma, bootstrap_value, apply_gradients=False):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        observations = np.array([row.tolist() for row in observations])
        next_observations = rollout[:, 1]
        next_observations = np.array([row.tolist() for row in next_observations])
        actions = rollout[:, 2]
        actions = np.array([row.tolist() for row in actions])
        rewards = rollout[:, 3]
        dones = rollout[:, 4]

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # rnn_state = self.local_AC.state_init
        target_feed_dict = {
            self.local_target_Q.state_holder: next_observations.reshape((-1, n_obs)),
        }
        target_next_Q = sess.run(self.local_target_Q.a_z, feed_dict=target_feed_dict)
        feed_dict = {
                     self.local_Q.next_Q: target_next_Q.reshape(-1, n_actions),
                     self.local_Q.r_holder: rewards.reshape(-1, 1, 1),
                     self.local_Q.state_holder: observations.reshape((-1, n_obs)),
                     self.local_Q.next_state_holder: next_observations.reshape((-1, n_obs)),
                     self.local_Q.action_holder: actions.reshape(-1, n_actions, 1),
                     self.local_Q.is_done_holder: dones.reshape(-1,),
                     }
        if apply_gradients:
            loss, _ = sess.run([self.local_Q.loss, self.local_Q.apply_grads], feed_dict=feed_dict)
        else:
            loss = sess.run(self.local_Q.loss, feed_dict=feed_dict)
        return loss / len(rollout)

    def work(self, max_episode_length, gamma, global_Q, sess, coord, saver):
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
                episode_error = 0
                episode_step_count = 0

                s = s1 = self.env.reset()
                episode_frames.append(s)
                s = pong_tools.prepro(s1, s)
                # rnn_state = self.local_AC.state_init
                done = False

                while not done:
                    # Take an action using probabilities from policy network output.
                    action_idx = policy(s, self.epsilon, self.local_target_Q)
                    a = action_list[action_idx]
                    # Calculating the one-hot action array for use by tensorflow
                    action_arr = np.zeros(n_actions)
                    action_arr[action_idx] = 1.

                    s1, r, done, info = self.env.step(a)
                    episode_frames.append(s1)
                    s1 = pong_tools.prepro(episode_frames[-1], episode_frames[-2]) if not done else s

                    episode_buffer.append([s, s1, action_arr, r, done])
                    # Update the network using the experience buffer at the end of the episode.
                    if total_steps % async_update_steps == 0:
                        loss = self.train(global_Q, episode_buffer, sess, gamma, 0.0, apply_gradients=True)
                        sess.run(self.update_local_ops)
                        # sess.run(self.local_Q.clear_grads)
                        self.local_Q.gradients = 0.
                        episode_buffer = []
                    elif r != 0 or (len(episode_buffer) > 0 and done):
                        loss = self.train(global_Q, episode_buffer, sess, gamma, 0.0)
                        episode_buffer = []

                    if total_steps % target_update_steps == 0:
                        sess.run(self.local_target_Q.update_target_network)

                    target_feed_dict = {
                        self.local_target_Q.state_holder: s1.reshape((-1, n_obs)),
                    }
                    target_next_Q = sess.run(self.local_target_Q.a_z, feed_dict=target_feed_dict)
                    err = sess.run(self.local_Q.loss, feed_dict={
                        self.local_Q.next_Q: target_next_Q.reshape(-1, n_actions),
                        self.local_Q.state_holder: s.reshape((-1, n_obs)),
                        self.local_Q.next_state_holder: s1.reshape((-1, n_obs)),
                        self.local_Q.action_holder: action_arr.reshape(-1, n_actions, 1),
                        self.local_Q.is_done_holder: np.array(done).reshape(-1,),
                        self.local_Q.r_holder: np.array(r).reshape(-1, 1, 1)})
                    episode_reward += r
                    episode_error += err
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_errs.append(episode_error/episode_step_count)

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                random_epsilon = np.absolute(np.random.normal(loc=0., scale=max(.2, (10 - episode_count)/10)))
                self.epsilon = random_epsilon if random_epsilon < 1. else 1.
                if episode_count % print_per_episode == 0:
                    print("{} {:<1.03}: in last {:<3} episodes before episode {:<4} avg REWARDS {:<2.04} +- {:<2.04}"
                          .format(self.name, self.epsilon, print_per_episode, episode_count,
                                  np.mean(self.episode_rewards[(episode_count - print_per_episode):episode_count]),
                                  np.std(self.episode_rewards[(episode_count - print_per_episode):episode_count])),
                          "avg STEPS",
                          np.mean(self.episode_lengths[(episode_count - print_per_episode):episode_count]), '+-',
                          np.std(self.episode_lengths[(episode_count - print_per_episode):episode_count]),
                          "bellman error",
                          np.mean(self.episode_errs[(episode_count - print_per_episode):episode_count]), '+-',
                          np.std(self.episode_errs[(episode_count - print_per_episode):episode_count]),
                          )

max_episode_length = s_size = a_size = 0
tf.reset_default_graph()


with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes',
                                  trainable=False)
    # trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    trainer = tf.train.RMSPropOptimizer(learning_rate)
    master_network = QNetwork(s_size, a_size, 'global', None)  # Generate global network
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
    sess.run(tf.initialize_all_variables())
    # sess.run(tf.global_variables_initializer())

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

