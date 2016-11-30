__author__ = "Anand Subramoney"

'''
    File name: ex3_deterministic.py
    Date created: 22/11/2016
    Date last modified: 24/11/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    Fill in the TODOs to implement policy gradient for ATARI 2600 pong with a learned baseline.
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
baseline_n_hidden = 10

state_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_obs), name='symbolic_state')
actions_one_hot_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_actions),
                                        name='symbolic_actions_one_hot_holder')
discounted_rewards_holder = tf.placeholder(dtype=tf.float32, shape=None, name='symbolic_reward')

with tf.name_scope('nonlinear_policy'):
    # TODO: Use the non-linear (neural network) implementation of policy gradient from the previous CartPole question
    # here but use 50 hidden neurons instead of 10!
    # Note that the dimensionality of observations and actions is different here from the CartPole problem.
    # NOTE: W_hid_var, W_var should be the names of the variables representing the weights to the hidden and output layer resp.
    #       bias_hid_var, bias_var should be the names of the variables representing the bias of the hidden and output neurons resp.
    #       These variables are used to calculate the baseline_loss below

    W_hid_var = tf.get_variable('W_hid_var', shape=(n_obs, n_hidden),
                              initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_obs)))
    bias_hid_var = tf.get_variable('bias_hid_var', shape=n_hidden, initializer=tf.constant_initializer(0.0))

    W_var = tf.get_variable('W_var', shape=(n_hidden, n_actions),
                            initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_obs)))
    bias_var = tf.get_variable('biasvar', shape=n_actions, initializer=tf.constant_initializer(0.0))

    # This list is used for calculating the gradients of the policy variables, and to minimize their variance using the
    # baseline.
    policy_variables = [W_hid_var, bias_hid_var, W_var, bias_var]

    # action_probabilies: dim = traj-length x n_actions, softmax over the output. Used for action selection in the
    # training loop
    a_y = tf.nn.bias_add(tf.matmul(state_holder, W_hid_var, name='output_activation'), bias_hid_var)
    y = tf.nn.tanh(a_y)
    a_z = tf.nn.bias_add(tf.matmul(y, W_var, name='output_activation'), bias_var)
    action_probabilities = tf.nn.softmax(a_z, name='action_probabilities')
    variable_summaries(action_probabilities, '/action_probabilities')


    # This operation is used for action selection during testing, to select the action with the maximum action probability
    testing_action_choice = tf.argmax(action_probabilities, dimension=1, name='testing_action_choice')


with tf.name_scope('baseline'):
    # NOTE: This implements a linear function for learning the baseline!
    # TODO: Replace this with a neural network with one hidden layer implemented as follows:
    # If the weights to your hidden layer and weights to the output layer are W_hid_baseline, and W_baseline resp.
    # and the biases of the hidden and output layer are bias_hid_baseline and bias_baseline resp.
    # * Then the activation of the hidden layer is given by:  a_y = W_hid_baseline x S + bias_hid_baseline. (S is the state)
    # * The output of the hidden layer is given by: y = relu(a_y). Use the Tensorflow function tf.nn.relu here
    # * The activation of the output layer is: a_z = W_baseline x y + bias_baseline
    # * The activation of the output layer is used as the baseline: baseline = a_z
    # Use 10 hidden neurons for the network

    # Implementation of linear function follows
    W_hid_baseline = tf.Variable(initial_value=np.random.rand(n_obs, baseline_n_hidden), validate_shape=True,
                                 trainable=True, dtype=tf.float32, name='hidden_theta_baseline')
    theta_var_baseline = tf.Variable(initial_value=np.random.rand(baseline_n_hidden, 1), validate_shape=True,
                                     trainable=True, dtype=tf.float32, name='theta_baseline')
    variable_summaries(theta_var_baseline, '/theta_baseline')
    bias_hid_baseline = tf.Variable(initial_value=np.zeros(baseline_n_hidden), validate_shape=True, trainable=True,
                                    dtype=tf.float32, name='hidden_bias_baseline')
    bias_var_baseline = tf.Variable(initial_value=np.zeros(1), validate_shape=True, trainable=True,
                                    dtype=tf.float32, name='bias_baseline')
    variable_summaries(bias_var_baseline, '/bias_baseline')

    # TODO: For the neural network implementation, add your hidden weights and biases to this list
    # This list is used for performing gradient descent on the baseline variables
    baseline_variables = [bias_var_baseline, theta_var_baseline, bias_hid_baseline, W_hid_baseline]

    a_y = tf.nn.bias_add(tf.matmul(state_holder, W_hid_baseline), bias_hid_baseline)
    y = tf.nn.relu(a_y)
    output_baseline = tf.nn.bias_add(tf.matmul(y, theta_var_baseline, name='output_activation'), bias_var_baseline)
    # output_baseline = tf.matmul(state_holder,
    #                             theta_var_baseline) + bias_var_baseline  # Dim -> trajectory length x 1
    variable_summaries(output_baseline, '/output_baseline')

    # Reshape n-trajectories x 1 2-D tensor to 1-D tensor
    baseline = tf.reshape(output_baseline, [-1])

    # We're using tf.stop_gradient here to make sure that minimizing L_theta doesn't update the baseline
    # We'll be updating the baseline separately based on the loss defined as the variance of the estimator
    baseline_no_grad = tf.stop_gradient(baseline)

with tf.name_scope('loss'):
    # TODO: Implement the loss function here
    # The loss function for policy gradient in Tensorflow is: sum(log_probabilities) * sum(discounted_rewards  - baseline_no_grad)
    # where the sum is over one trajectory. The log probabilities are the log of probability of current action taken
    # i.e. log(\pi(a_t|s_t))
    #
    # * First calculate the action probabilities of the actions that were *actually chosen* as follows:
    #   Multiply the one-hot vectors passed in from the main loop through the 'actions_one_hot_holder' placeholder
    #   with the action probabilities calculated above.
    # * Then you can use the tf.reduce_sum function to sum over the trajectory to calculate each of the terms
    #   sum(log_probabilities) and sum(discounted_rewards - baseline_no_grad)
    # * Note that you have to use the variable baseline_no_grad here so that minimizing this loss function doesn't
    #   change the baseline parameters.
    # * The final loss function is just a product of the above two sums.
    # * Note that for calculating the loss, the *placeholders* passed in all have a first dimension equal to
    #   'length of trajectory'.
    # * For each variable/operation/placeholder, remember to call the 'variable_summaries' function to enable recording
    #   of the the variable for tensorboard to visualize
    # NOTE: 'log_probability_sum' should be the name of the operation representing sum(log_probabilities)
    #       'discounted_rewards_sum' should be the name of the operation representing sum(discounted_rewards - baseline_no_grad)
    #       Both these variables are used below to calculate the baseline loss.

    chosen_action_prob = tf.batch_matmul(tf.reshape(action_probabilities, (-1, 1, n_actions)),
                                         tf.reshape(actions_one_hot_holder, (-1, n_actions, 1)))
    variable_summaries(chosen_action_prob, '/chosen_action_prob')
    log_probability_sum = tf.reduce_sum(tf.log(chosen_action_prob))
    discounted_rewards_sum = tf.reduce_sum(tf.sub(discounted_rewards_holder, baseline_no_grad))
    # Call your final loss function L_theta (This is used below in the gradient descent step).
    # Remember to add a -ve sign since we want to maximize this, but tensorflow has only the minimize operation.
    # Note: This is a scalar.
    L_theta = - log_probability_sum * discounted_rewards_sum
    variable_summaries(L_theta, '/L_theta')

with tf.name_scope('train'):
    # We define the optimizer to use the RMSProp optimizer, and ask it to minimize our loss
    grad_apply = tf.train.RMSPropOptimizer(learning_rate).minimize(L_theta)

with tf.name_scope('baseline_loss'):
    gradients_holder = tf.placeholder(dtype=tf.float32, shape=(None, None), name='symbolic_L_theta_gradients')
    # We define L_theta again that uses the real baseline with the trainable variable
    # The actual updates to the baseline parameters will be the gradients of the gradients of L_theta w.r.t to the
    # baseline parameters
    discounted_rewards_sum_for_baseline = tf.reduce_sum(discounted_rewards_holder - baseline, reduction_indices=0,
                                                        name='discounted_rewards_sum_for_baseline')
    L_theta_for_baseline = -(log_probability_sum * discounted_rewards_sum_for_baseline)

    # Calculate the gradients w.r.t to the relevant variables
    L_theta_gradient_for_baseline = tf.gradients(L_theta_for_baseline, policy_variables)

    # Unpack all the variables so that we get one tensor that's a horizontal stack of the gradients
    L_theta_gradient_for_baseline_tensor = tf.concat(0, [tf.reshape(g, [-1]) for g in L_theta_gradient_for_baseline])

    # Calculate the mean of the gradients passed in using gradients_holder. This is the expectation of the gradients
    L_theta_gradient_for_baseline_tensor_mean = tf.reduce_mean(gradients_holder, reduction_indices=0)

    # Define the loss here as the variance of our gradient estimator
    # i.e. if \theta_est is our estimator, then we want to minimize E[(\theta_est - E(\theta_est))^2]
    # Where E[.] refers to the expectation
    baseline_loss = tf.reduce_mean(
        (L_theta_gradient_for_baseline_tensor - L_theta_gradient_for_baseline_tensor_mean) ** 2)
    variable_summaries(baseline_loss, '/baseline_loss')
    baseline_opt = tf.train.RMSPropOptimizer(0.001).minimize(baseline_loss, var_list=baseline_variables)

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
gradients_list = []  # List of gradients we need to train the baseline

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
        # First calculate the gradients that'll be used for learning the baseline
        # This returns a tensor with all the gradients concatenated into one Tensor
        # We accumulate these gradients over the entire experiment to be able to calculate it's mean
        gradients = sess.run(L_theta_gradient_for_baseline_tensor,
                             feed_dict={actions_one_hot_holder: action_one_hots, state_holder: states,
                                        discounted_rewards_holder: batch_rewards})
        gradients_list.append(gradients)

        # Then calculate the discounted rewards for each step
        batch_reward_length = len(batch_rewards)
        discounted_batch_rewards = batch_rewards.copy()
        for i in range(batch_reward_length):
            discounted_batch_rewards[i] *= (gamma ** (batch_reward_length - i - 1))

        # Just perform the gradient descent step for the main loss function L_theta
        sess.run(grad_apply,
                 feed_dict={actions_one_hot_holder: action_one_hots, state_holder: states,
                            discounted_rewards_holder: discounted_batch_rewards})

        # Now we update the baseline parameters, by passing it the list of gradients stored
        # so that it can find it's mean and calculate the variance of the gradients.
        # Remember that we're trying to minimize the variance of the gradient estimates by learning this baseline.
        summary, _ = sess.run([merged, baseline_opt],
                              feed_dict={gradients_holder: np.array(gradients_list),
                                         actions_one_hot_holder: action_one_hots,
                                         state_holder: states,
                                         discounted_rewards_holder: batch_rewards})
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
            print(
                "Episode {}: Average STEPS in last {} episodes before episode {}".format(episode_no, print_per_episode),
                np.mean(episode_steps_list[(episode_no - print_per_episode):episode_no]), '+-',
                np.std(episode_steps_list[(episode_no - print_per_episode):episode_no])
            )

print("Average REWARDS in the 100 test steps: {:.2f}+-{.2f}".format(np.mean(episode_rewards_list),
                                                                    np.std(episode_rewards_list)))
print("Average STEPS in the 100 test steps: {:.2f}+-{.2f}".format(np.mean(episode_steps_list),
                                                                  np.std(episode_steps_list)))

ax = plt.subplot(122)
ax.plot(range(len(episode_rewards_list)), episode_rewards_list)
ax.set_title("Test")
ax.set_xlabel('Episode number')
ax.set_ylabel('Episode reward')
plt.show()
