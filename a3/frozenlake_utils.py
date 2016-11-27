import numpy as np

import matplotlib.pyplot as plt


def plot_policy(ax, env, Q_table):
    n_state, n_action = Q_table.shape

    def to_map(c):
        if c == b'S':
            return 2
        elif c == b'F':
            return 1
        elif c == b'H':
            return -2
        else:
            return 0

    map0 = np.vectorize(to_map)(env.desc)

    def check_terminal(c):
        if c in [b'F', b'S']:
            return 0
        else:
            return 1

    is_terminal = np.vectorize(check_terminal)(env.desc)

    start_position = np.concatenate(np.where(env.desc == b'S'))
    goal_position = np.concatenate(np.where(env.desc == b'G'))
    assert start_position.shape == (2,), 'Weird start position {}'.format(goal_position)
    assert goal_position.shape == (2,), 'Weird goal position {}'.format(goal_position)

    is_terminal = np.array(is_terminal, dtype=bool)
    map0 = np.array(map0)

    ax.pcolormesh(np.arange(env.ncol + 1), np.arange(env.nrow + 1), map0)

    def next_state(i, j, a):
        if a == 0:
            return i, j - 1
        elif a == 1:
            return i + 1, j
        elif a == 2:
            return i, j + 1
        elif a == 3:
            return i - 1, j
        else:
            raise ValueError('Unkown action {}'.format(a))

    ax.set_ylim([env.ncol, 0])

    for s in range(n_state):
        i, j = s // env.nrow, np.mod(s, env.ncol)
        if not (is_terminal[i, j]):
            a = np.argmax(Q_table[s, :])
            i_, j_ = next_state(i, j, a)
            ax.arrow(j + .5, i + .5, (j_ - j) * .8, (i_ - i) * .8, head_width=.1, color='white')

    ax.annotate('S', xy=(start_position[0] + .35, start_position[1] + .65), color='black', size=20)
    ax.annotate('G', xy=(goal_position[0] + .35, goal_position[1] + .65), color='black', size=20)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Policy')


def plot_results(n_train_trials, n_test_trials, reward_list, Q_table, env):
    reward_list = np.array(reward_list)
    un_r = np.unique(reward_list)
    assert (un_r == [0, 1]).all() or (un_r == [0]).all() or (un_r == [1]).all(), '''
    The reward list should be only 0 and 1 for the FrozenLake problem.
    If you transform the reward in the algorithm, the natural reward [0,1] should still be reported.
    Now we found {}'''.format(np.unique(reward_list))

    fig, ax_list = plt.subplots(2, 2)

    # plot the reward over time
    ax_list[0, 0].plot(reward_list[:n_train_trials], lw=2)
    ax_list[0, 0].set_ylim([0, 1.1])
    ax_list[0, 0].set_xticks([0, n_train_trials / 2, n_train_trials])
    ax_list[0, 0].set_title('Acc. reward - Training')
    ax_list[0, 0].set_ylabel('Accumulated reward')
    ax_list[0, 0].set_xlabel('Trial number')

    # plot the reward over time
    r_av = np.mean(reward_list[n_train_trials:])
    ax_list[0, 1].plot(reward_list[n_train_trials:], lw=2)
    ax_list[0, 1].set_ylim([0, 1.1])
    ax_list[0, 1].axhline(y=r_av, color='red', lw=1)
    ax_list[0, 1].set_xticks([0, n_test_trials])
    ax_list[0, 1].set_yticks([0, r_av, 1])
    ax_list[0, 1].set_ylim([0, 1.1])
    ax_list[0, 1].set_title('Acc. reward - Testing')
    ax_list[0, 1].set_xlabel('Trial number')

    # plot the policy
    plot_policy(ax_list[1, 0], env, Q_table)

    start_position = np.concatenate(np.where(env.desc == b'S'))
    goal_position = np.concatenate(np.where(env.desc == b'G'))
    assert start_position.shape == (2,), 'Weird start position {}'.format(goal_position)
    assert goal_position.shape == (2,), 'Weird goal position {}'.format(goal_position)

    # Plot the value function
    V = np.max(Q_table, axis=1).reshape(env.ncol, env.nrow)
    im = ax_list[1, 1].imshow(V, interpolation='nearest', vmin=np.floor(V.min()), vmax=np.ceil(V.max()))
    ax_list[1, 1].set_xticks([])
    ax_list[1, 1].set_yticks([])
    ax_list[1, 1].set_title('Value fun. (from Q table)')
    ax_list[1, 1].annotate('S', xy=(start_position[0] - .15, start_position[0] + .15), color='white', size=20)
    ax_list[1, 1].annotate('G', xy=(goal_position[0] - .15, goal_position[0] + .15), color='white', size=20)

    cbar_ax = fig.add_axes([0.925, 0.05, 0.025, 0.3])
    fig.colorbar(im, cax=cbar_ax, ticks=[np.floor(V.min()), np.ceil(V.max())])
    plt.tight_layout()


def plot_variation_of_perf_vs_lmbda(lmbdas, mean_test_rewards):
    plt.figure()

    plt.plot(lmbdas, mean_test_rewards)

    plt.xlabel("Lambdas")
    plt.ylabel("Mean test rewards")
    plt.title("Variation of test reward with lambda")

    plt.tight_layout()
