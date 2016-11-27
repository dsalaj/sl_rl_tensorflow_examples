__author__ = "Anand Subramoney"

'''
    File name: ex2_cartpole_policy_gradient.py
    Date created: 22/11/2016
    Date last modified: 24/11/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    Fill in the TODOs to implement policy gradient with a linear (ex 2a) or non-linear (ex 2b) policy for the
    CartPole environment
'''

import time

import numpy as np
import tensorflow as tf
from gym.envs.classic_control import CartPoleEnv

import matplotlib.pyplot as plt
from tf_tools import variable_summaries, parameter_summaries

env = CartPoleEnv()
should_render = False

n_actions = env.action_space.n
n_obs = env.observation_space.high.__len__()

print_per_episode = 100

max_episode_len = 200
batch_size = 200
n_train_trials = 1000
n_test_trials = 100
gamma = 0.9999
learning_rate = 0.01

constant_baseline = 1.

# Use parameter_summaries and variable_summaries functions to record data for tensorboard
parameter_summaries([batch_size, n_train_trials, n_test_trials, gamma, learning_rate],
                    ['batch_size', 'n_train_trials', 'n_test_trials', 'gamma', 'learning_rate'])

# The first dimension being defined as None means that tensorflow will infer the shape for us.
# The first dimension is the length of the trajectory which can vary with each update.
state_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_obs), name='symbolic_state')
actions_one_hot_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_actions),
                                        name='symbolic_actions_one_hot_holder')

discounted_rewards_holder = tf.placeholder(dtype=tf.float32, shape=None, name='symbolic_reward')

# Tensorflow name_scope allows us to group values for use in Tensorboard
with tf.name_scope('linear_policy'):
    # TODO: Implement the linear policy here for exercise 2a
    # (For exercise 2b, replace the below with a non-linear policy as described in the assignment)
    # The policy i.e. the probability distribution of actions for each state is given by a softmax over the output of
    #   \theta^T x S + b where \theta is the parameter vector and S is the state vector and b is the bias vector.
    # * Define theta and bias as Tensorflow variables
    # * Calculate the  output of the linear model as: \theta^T x S + b
    # * Calculate the action probabilities as the softmax over the output (using tf.nn.softmax function)
    # * These action probabilities are used for selecting the actions during training (in the main loop)

    ...

    # action_probabilies: dim = traj-length x n_actions, softmax over the output. Used for action selection in the
    # training loop
    action_probabilities = ...
    variable_summaries(action_probabilities, '/action_probabilities')

    # This operation is used for action selection during testing, to select the action with the maximum action probability
    testing_action_choice = tf.argmax(action_probabilities, dimension=1, name='testing_action_choice')

with tf.name_scope('loss'):
    # TODO: Implement the loss function here
    # The loss function for policy gradient in Tensorflow is: sum(log_probabilities) * sum(discounted_sum_of_rewards)
    # where the sum is over one trajectory. The log probabilities are the log of probability of current action taken
    # i.e. log(\pi(a_t|s_t))
    #
    # * First calculate the action probabilities of the actions that were *actually chosen* as follows:
    #   Multiply the one-hot vectors passed in from the main loop through the 'actions_one_hot_holder' placeholder
    #   with the action probabilities calculated above.
    # * Then you can use the tf.reduce_sum function to sum over the trajectory to calculate each of the terms
    #   sum(log_probabilities) and sum(discounted sum of rewards)
    # * The final loss function is just a product of the above two sums.
    # * Note that for calculating the loss, the placeholders passed in all have a first dimension equal to
    #   'length of trajectory'.
    # * For each variable/operation/placeholder, remember to call the 'variable_summaries' function to enable recording
    #   of the the variable for tensorboard to visualize

    ...

    # Call your final loss function L_theta (This is used below in the gradient descent step).
    # Remember to add a -ve sign since we want to maximize this, but tensorflow has only the minimize operation.
    # Note: This is a scalar.
    L_theta = ...
    variable_summaries(L_theta, '/L_theta')

with tf.name_scope('train'):
    # We define the optimizer to use the ADAM optimizer, and ask it to minimize our loss
    gd_opt = tf.train.AdamOptimizer(learning_rate).minimize(L_theta)

sess = tf.Session()  # FOR NOW everything is symbolic, this object has to be called to compute each value of Q

# SUMMARIES
# Let's setup the tensorboard stuff to write to a directory called tensorboard
merged = tf.merge_all_summaries()
suffix = time.strftime('%Y-%m-%d--%H-%M-%S')
train_writer = tf.train.SummaryWriter('tensorboard/cartpole/{}'.format(suffix) + '/train', sess.graph)
test_writer = tf.train.SummaryWriter('tensorboard/cartpole/{}'.format(suffix) + '/test')
# Start

sess.run(tf.initialize_all_variables())

observation = env.reset()
batch_rewards = []
states = []
action_one_hots = []

episode_rewards = []
episode_rewards_list = []
episode_steps_list = []

step = 0
episode_no = 0
while episode_no <= n_train_trials:
    if should_render: env.render()
    step += 1

    action_probability_values = sess.run(action_probabilities,
                                         feed_dict={state_holder: [observation]})
    # Choose the action using the action probabilities output by the policy implemented in tensorflow.
    action = np.random.choice(np.arange(n_actions), p=action_probability_values.ravel())

    # Calculating the one-hot action array for use by tensorflow
    action_arr = np.zeros(n_actions)
    action_arr[action] = 1.
    action_one_hots.append(action_arr)

    # Record states
    states.append(observation)

    observation, reward, done, info = env.step(action)
    # We don't want to go above 200 steps
    if step >= 200:
        done = True

    batch_rewards.append(reward)
    episode_rewards.append(reward)

    # If the episode is done, and it contained at least one step, do the gradient updates
    if len(batch_rewards) > 0 and done:

        # First calculate the discounted rewards for each step
        batch_reward_length = len(batch_rewards)
        discounted_batch_rewards = batch_rewards.copy()
        for i in range(batch_reward_length):
            discounted_batch_rewards[i] *= (gamma ** (batch_reward_length - i - 1))

        # Next run the gradient descent step
        # Note that each of action_one_hots, states, discounted_batch_rewards has the first dimension as the length
        # of the current trajectory
        summary, gradients = sess.run([merged, gd_opt],
                                      feed_dict={actions_one_hot_holder: action_one_hots, state_holder: states,
                                                 discounted_rewards_holder: discounted_batch_rewards})
        train_writer.add_summary(summary, episode_no)

        action_one_hots = []
        states = []
        batch_rewards = []

    if done:
        # Done with episode. Reset stuff.
        episode_no += 1

        episode_rewards_list.append(np.sum(episode_rewards))
        episode_steps_list.append(step)

        episode_rewards = []

        step = 0

        observation = env.reset()

        if episode_no % print_per_episode == 0:
            print("Episode {}: Average steps in last {} episodes".format(episode_no, print_per_episode),
                  np.mean(episode_steps_list[(episode_no - print_per_episode):episode_no]), '+-',
                  np.std(episode_steps_list[(episode_no - print_per_episode):episode_no])
                  )

plt.figure()
ax = plt.subplot(121)
ax.plot(range(len(episode_rewards_list)), episode_rewards_list)
ax.set_title("Training rewards")
ax.set_xlabel('Episode number')
ax.set_ylabel('Episode reward')

observation = env.reset()

episode_rewards_list = []
episode_rewards = []
episode_steps_list = []

step = 0
episode_no = 0

print("Testing")
while episode_no <= n_test_trials:
    env.render()
    step += 1

    # For testing, we choose the action using an argmax.
    test_action, = sess.run([testing_action_choice],
                            feed_dict={state_holder: [observation]})

    observation, reward, done, info = env.step(test_action[0])
    if step >= 200:
        done = True
    episode_rewards.append(reward)

    if done:
        episode_no += 1

        episode_rewards_list.append(np.sum(episode_rewards))
        episode_steps_list.append(step)

        episode_rewards = []
        step = 0
        observation = env.reset()

        if episode_no % print_per_episode == 0:
            print("Episode {}: Average steps in last {} episodes".format(episode_no, print_per_episode),
                  np.mean(episode_steps_list[(episode_no - print_per_episode):episode_no]), '+-',
                  np.std(episode_steps_list[(episode_no - print_per_episode):episode_no])
                  )

ax = plt.subplot(122)
ax.plot(range(len(episode_rewards_list)), episode_rewards_list)
ax.set_title("Test")
ax.set_xlabel('Episode number')
ax.set_ylabel('Episode reward')
plt.show()
