__author__ = "Anand Subramoney"

'''
    File name: ex3_deterministic.py
    Date created: 22/11/2016
    Date last modified: 24/11/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    Fill in the TODOs to implement policy gradient for ATARI 2600 pong
'''


import time

import numpy as np
import tensorflow as tf

import gym

import matplotlib.pyplot as plt
from pong_tools import prepro
from tf_tools import variable_summaries

env = gym.make("Pong-v0")

should_render = False
# 0-> no movement, 2->UP, 3->DOWN
n_actions = 3
action_list = [0, 2, 3]
# The *relative* y coordinate of the opponent and the x,y coordinates of the ball for *two* frames
n_obs = 6

print_per_episode = 100

n_train_trials = 1000
n_test_trials = 100
gamma = 0.9999
learning_rate = 0.001

n_hidden = 50

state_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_obs), name='symbolic_state')
actions_one_hot_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_actions),
                                        name='symbolic_actions_one_hot_holder')
discounted_rewards_holder = tf.placeholder(dtype=tf.float32, shape=None, name='symbolic_reward')

with tf.name_scope('nonlinear_policy'):
    # TODO: Use the non-linear (neural network) implementation of policy gradient from the previous CartPole question
    # here but use 50 hidden neurons instead of 10!
    # Note that the dimensionality of observations and actions is different here from the CartPole problem.

    theta_h = tf.get_variable('theta_h', shape=(n_obs, n_hidden),
                              initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_obs)))
    b_h = tf.get_variable('b_h', shape=n_hidden, initializer=tf.constant_initializer(0.0))

    theta = tf.get_variable('theta', shape=(n_hidden, n_actions),
                            initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_obs)))
    b = tf.get_variable('b', shape=n_actions, initializer=tf.constant_initializer(0.0))

    # action_probabilies: dim = traj-length x n_actions, softmax over the output. Used for action selection in the
    # training loop
    a_y = tf.matmul(state_holder, theta_h, name='output_activation') + b_h
    y = tf.nn.tanh(a_y)
    a_z = tf.matmul(y, theta, name='output_activation') + b
    action_probabilities = tf.nn.softmax(a_z, name='action_probabilities')
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

    # chosen_action_prob = tf.batch_matmul(tf.reshape(action_probabilities, (-1, 1, n_actions)),
    #                                      tf.reshape(actions_one_hot_holder, (-1, n_actions, 1)))
    chosen_action_prob = tf.reduce_sum(action_probabilities * tf.reshape(actions_one_hot_holder, (-1, n_actions)), 1)
    variable_summaries(chosen_action_prob, '/chosen_action_prob')
    # TODO: ASK: There are TODOs to implement the policy, the loss and in the main loop to calculate the discounted sum of rewards
    #       ASK: Traceback (most recent call last):
    #              File "ex3_pong_policy_gradient_constant_baseline.py", line 244, in <module>
    #                "Episode {}: Average STEPS in last {} episodes before episode {}".format(episode_no, print_per_episode),
    #            IndexError: tuple index out of range


    # Call your final loss function L_theta (This is used below in the gradient descent step).
    # Remember to add a -ve sign since we want to maximize this, but tensorflow has only the minimize operation.
    # Note: This is a scalar.
    L_theta = - tf.reduce_sum(tf.log(chosen_action_prob)) * tf.reduce_sum(discounted_rewards_holder)
    variable_summaries(L_theta, '/L_theta')

with tf.name_scope('train'):
    # We define the optimizer to use the RMSProp optimizer, and ask it to minimize our loss
    grad_apply = tf.train.RMSPropOptimizer(learning_rate).minimize(L_theta)

sess = tf.Session()  # FOR NOW everything is symbolic, this object has to be called to compute each value of Q

# SUMMARIES
merged = tf.merge_all_summaries()
suffix = time.strftime('%Y-%m-%d--%H-%M-%S')
train_writer = tf.train.SummaryWriter('tensorboard/pong/{}'.format(suffix) + '/train', sess.graph)
test_writer = tf.train.SummaryWriter('tensorboard/pong/{}'.format(suffix) + '/test')
# Start

sess.run(tf.initialize_all_variables())

prev_observation = observation = env.reset()
batch_rewards = []
states = []
action_one_hots = []

single_episode_rewards = []
episode_rewards_list = []
episode_steps_list = []

step = 0
episode_no = 0
while episode_no <= n_train_trials:
    if should_render: env.render()
    step += 1

    # Preprocess the observation (which is an image) before using it to select an action.
    processed_observation = prepro(observation, prev_observation)

    # Choose the action using the action probabilities output by the policy implemented in tensorflow.
    action_probability_values = sess.run(action_probabilities,
                                         feed_dict={state_holder: processed_observation.reshape(1, n_obs)})
    action_idx = np.random.choice(range(n_actions), p=action_probability_values.ravel())
    action = action_list[action_idx]

    # Calculating the one-hot action array for use by tensorflow
    action_arr = np.zeros(n_actions)
    action_arr[action_idx] = 1.
    action_one_hots.append(action_arr)

    # Record states
    states.append(processed_observation)

    prev_observation = observation
    observation, reward, done, info = env.step(action)

    batch_rewards.append(reward)
    single_episode_rewards.append(reward)

    if reward != 0 or (len(batch_rewards) > 0 and done):
        # First calculate the discounted rewards for each step
        batch_reward_length = len(batch_rewards)
        discounted_batch_rewards = batch_rewards.copy()
        for i in range(batch_reward_length):
            discounted_batch_rewards[i] *= (gamma ** (batch_reward_length - i - 1))

        # Next run the gradient descent step
        # Note that each of action_one_hots, states, discounted_batch_rewards has the first dimension as the length
        # of the current trajectory
        summary, _ = sess.run([merged, grad_apply],
                              feed_dict={actions_one_hot_holder: action_one_hots, state_holder: states,
                                         discounted_rewards_holder: discounted_batch_rewards})
        train_writer.add_summary(summary, episode_no)

        action_one_hots = []
        states = []
        batch_rewards = []

    if done:
        # Done with episode. Reset stuff.
        episode_no += 1

        episode_rewards_list.append(np.sum(single_episode_rewards))
        episode_steps_list.append(step)

        single_episode_rewards = []

        step = 0

        observation = env.reset()

        if episode_no % print_per_episode == 0:
            print("Average REWARDS in last {} episodes before episode {}".format(print_per_episode, episode_no),
                  np.mean(episode_rewards_list[(episode_no - print_per_episode):episode_no]), '+-',
                  np.std(episode_rewards_list[(episode_no - print_per_episode):episode_no])
                  )
            print("Average STEPS in last {} episodes before episode {}".format(print_per_episode, episode_no),
                  np.mean(episode_steps_list[(episode_no - print_per_episode):episode_no]), '+-',
                  np.std(episode_steps_list[(episode_no - print_per_episode):episode_no])
                  )

plt.figure()
ax = plt.subplot(121)
ax.plot(range(len(episode_rewards_list)), episode_rewards_list)
ax.set_title("Training rewards")
ax.set_xlabel('Episode number')
ax.set_ylabel('Episode reward')

prev_observation = observation = env.reset()

episode_rewards_list = []
single_episode_rewards = []
episode_steps_list = []

step = 0
episode_no = 0

print("Testing")
while episode_no <= n_test_trials:
    if should_render: env.render()
    step += 1

    processed_observation = prepro(observation, prev_observation)
    # For testing, we choose the action using an argmax.
    test_action_idx, = sess.run([testing_action_choice],
                                feed_dict={state_holder: processed_observation.reshape(1, n_obs)})
    test_action = action_list[test_action_idx[0]]

    prev_observation = observation
    observation, reward, done, info = env.step(test_action)
    single_episode_rewards.append(reward)

    if done:
        episode_no += 1

        episode_rewards_list.append(np.sum(single_episode_rewards))
        episode_steps_list.append(step)

        single_episode_rewards = []
        step = 0
        observation = env.reset()

        if episode_no % print_per_episode == 0:
            print("Episode {}: Average REWARDS in last {} episodes".format(episode_no, print_per_episode),
                  np.mean(episode_rewards_list[(episode_no - print_per_episode):episode_no]), '+-',
                  np.std(episode_rewards_list[(episode_no - print_per_episode):episode_no])
                  )
            # print(
            #     "Episode {}: Average STEPS in last {} episodes before episode {}".format(episode_no, print_per_episode),
            #     np.mean(episode_steps_list[(episode_no - print_per_episode):episode_no]), '+-',
            #     np.std(episode_steps_list[(episode_no - print_per_episode):episode_no])
            # )

print("Average REWARDS in the 100 test steps: {:.2f}+-{:.2f}".format(np.mean(episode_rewards_list),
                                                                    np.std(episode_rewards_list)))
# print("Average STEPS in the 100 test steps: {:.2f}+-{.2f}".format(np.mean(episode_steps_list),
#                                                                   np.std(episode_steps_list)))

ax = plt.subplot(122)
ax.plot(range(len(episode_rewards_list)), episode_rewards_list)
ax.set_title("Test")
ax.set_xlabel('Episode number')
ax.set_ylabel('Episode reward')
plt.show()
train_writer.close()
test_writer.close()
