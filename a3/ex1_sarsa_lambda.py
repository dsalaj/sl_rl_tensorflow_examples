__author__ = "Anand Subramoney"

'''
    File name: ex1_frozen_lake_template.py
    Date created: 22/11/2016
    Date last modified: 24/11/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    Use this template to implement and analyse SARSA-lambda and Watkin's Q-lambda for the
    Frozen Lake environment
'''

from frozenlake_utils import plot_results, plot_variation_of_perf_vs_lmbda

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

# Generate the environment
env = FrozenLakeEnv(map_name='4x4', is_slippery=True)


def simulate(lmbda):
    # Algorithm parameters
    should_render = False
    should_plot = False
    should_print_training_updates = False

    learning_rate = .1
    lr_decay = 0.9999

    epsilon = 1.
    epsilon_decay = .999

    gamma = .9

    n_train_trials = 2000
    n_test_trials = 500
    trial_duration = 1000

    # lmbda = 0.3

    n_state = env.observation_space.n
    n_action = env.action_space.n

    # Initialize the Q values
    Q_table = np.zeros((n_state, n_action), dtype=np.float)
    eligibility_traces = np.zeros((n_state, n_action), dtype=np.float)

    def policy(Q_table, state, epsilon):
        '''
        This should implement an epsilon greedy policy:
        - with probability epsilon take a random action
        - otherwise take RANDOMLY an action the action that maximizes Q(s,a) with s the current state

        :param Q_table:
        :param state:
        :return:
        '''

        if rd.uniform() < epsilon:
            action = rd.randint(low=0, high=n_action)
        else:
            max_actions = np.argwhere(Q_table[state] == np.amax(Q_table[state])).flatten()
            action = rd.choice(max_actions)
        return action

    def update_Q_table(Q_table, state, action, reward, new_state, new_action, is_done):
        '''
        Update the Q values according to the SARSA(\lambda) algorithm.
        Also update the eligibility traces here
        if \lambda = 0, then the update should be equivalent to 1-step SARSA

        :param Q_table: Table of expected return for each pair of action and state
        :param state:
        :param action:
        :param reward:
        :param new_state:
        :param new_action:
        :param is_done:
        :return:
        '''

        # TODO: Implement the update of the Q table and the eligibility traces according to
        # SARSA-lambda or Watkin's Q-lambda (both algorithms described in the practical slides)
        # Note: If the trial stops, make sure to set 0 as the estimation of all future return

        if is_done:
            d = (reward - Q_table[state, action])
        else:
            d = (reward + gamma * Q_table[new_state, new_action] - Q_table[state, action])

        # reference: http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node77.html
        eligibility_traces[state, action] += 1

        for s in range(Q_table.shape[0]):
            for a in range(Q_table.shape[1]):
                Q_table[s, a] += learning_rate * d * eligibility_traces[s, a]
                eligibility_traces[s, a] *= gamma * lmbda

    reward_list = []
    for k in range(n_train_trials + n_test_trials):
        acc_reward = 0  # Init the accumulated reward
        observation = env.reset()  # Init the state
        action = policy(Q_table, observation, epsilon)  # Init the first action
        # resetting eligibility traces
        eligibility_traces = np.zeros((n_state, n_action), dtype=np.float)

        for t in range(trial_duration):  # The number of time steps in this game is maximum 200
            if should_render: env.render()

            #####################
            # TODO:
            # In the APPROPRIATE order do the following:
            #   - Update the Q table with the function update_Q_table()
            #   - choose an action with the function policy()
            #   - perform an action
            #
            # TIP:
            #   Think about the final state the specific case just before getting the reward for the first time.
            #   The "goodness" of the rewarding state has to propagate to the previous one.

            new_observation, reward, done, info = env.step(action)  # Take the action
            new_action = policy(Q_table, new_observation, epsilon)

            if k < n_train_trials:
                update_Q_table(Q_table, observation, action, reward, new_observation, new_action, done)

            observation = new_observation  # Pass the new state to the next step
            action = new_action  # Pass the new action to the next step
            acc_reward += reward  # Accumulate the reward
            if done:
                if (k + 1) < n_train_trials:
                    epsilon *= epsilon_decay  # Decaying epsilon
                    learning_rate *= lr_decay  # Decaying learning rate
                else:
                    epsilon = 0  # Stop epsilon during testing
                if should_print_training_updates and k % 100 == 0:
                    print("After", k, " trials, ", "reward for last 100 trials is ", np.mean(reward_list[-100:]), "+-",
                          np.std(reward_list[-100:]))
                break  # Stop the trial when the environment says it is done

        reward_list.append(acc_reward)  # Store the result

    mean_test_reward = np.mean(reward_list[n_train_trials:])
    std_test_reward = np.std(reward_list[n_train_trials:])
    print('Average accumulated reward in {} test runs: {:.3g} +- {:.3g}'.format(n_test_trials, mean_test_reward,
                                                                                std_test_reward))
    if should_plot:
        plot_results(n_train_trials, n_test_trials, reward_list, Q_table, env)
        plt.show()

    return mean_test_reward


def main():
    lmbdas = np.arange(0, 1.1, 0.1)
    mean_test_rewards = []
    for lmbda in lmbdas:
        mean_test_reward = simulate(lmbda=lmbda)
        mean_test_rewards.append(mean_test_reward)
    plot_variation_of_perf_vs_lmbda(lmbdas, mean_test_rewards)
    plt.show()


if __name__ == '__main__':
    main()
