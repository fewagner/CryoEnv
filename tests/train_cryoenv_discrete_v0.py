import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm.auto import tqdm, trange
from functools import partial
import multiprocessing as mp
# from IPython.display import clear_output

import warnings

warnings.simplefilter('ignore')
np.random.seed(0)
gym.logger.set_level(40)


def train_agent(agent,
                env,
                Q,
                training_rewards,
                epsilons,
                train_episodes,
                max_steps,
                epsilon,
                alpha,
                discount_factor,
                min_epsilon,
                max_epsilon,
                decay,
                multiprocessing,
                save_q_after_ep,
                save_q_for_agent,
                ):
    if multiprocessing:
        iterator = range
        Q[:] = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
    else:
        iterator = trange

    for episode in iterator(train_episodes):
        # Reseting the environment each time as per requirement
        state = env.reset()
        # Starting the tracker for the rewards
        total_training_rewards = 0

        for step in range(max_steps):
            # Choosing an action given the states based on a random number
            exp_exp_tradeoff = random.uniform(0, 1)

            ### STEP 2: SECOND option for choosing the initial action - exploit
            # If the random number is larger than epsilon: employing exploitation
            # and selecting best action
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state, :])

            ### STEP 2: FIRST option for choosing the initial action - explore
            # Otherwise, employing exploration: choosing a random action
            else:
                action = env.action_space.sample()

            ### STEPs 3 & 4: performing the action and getting the reward
            # Taking the action and getting the reward and outcome state
            new_state, reward, done, info = env.step(action)

            ### STEP 5: update the Q-table
            # Updating the Q-table using the Bellman equation
            Q[state, action] = Q[state, action] + alpha * (
                    reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])
            # Increasing our total reward and updating the state
            total_training_rewards += reward
            state = new_state

            # Ending the episode
            if done == True:
                # print ("Total reward for episode {}: {}".format(episode, total_training_rewards))
                break

        # Cutting down on exploration by reducing the epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        # Adding the total reward and reduced epsilon values
        training_rewards[agent, episode] = total_training_rewards / max_steps
        epsilons[agent, episode] = epsilon

        # save Q table
        if agent in save_q_for_agent and episode in save_q_after_ep:
            np.save("q_table_ep" + str(episode) + "_ag" + str(agent), Q)


def test_agent(env,
               Q,
               steps,
               ):
    state = env.reset()

    for step in range(steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        state = new_state

        if done == True:
            break


if __name__ == '__main__':

    # ------------------------------------------------
    # CREATE THE ENVIRONMENT
    # ------------------------------------------------

    # constants
    env_kwargs = {
        'V_set_iv': (0, 99),
        'V_set_step': 1,
        'ph_iv': (0, 0.99),
        'ph_step': 0.01,
        'wait_iv': (10, 50),
        'wait_step': 2,
        'heater_resistance': np.array([100.]),
        'thermal_link_channels': np.array([[1.]]),
        'thermal_link_heatbath': np.array([1.]),
        'temperature_heatbath': 0.,
        'min_ph': 0.1,
        'g': np.array([0.0001]),
        'T_hyst': np.array([0.001]),
        'control_pulse_amplitude': 50,
        'env_fluctuations': 0.005,
        'save_trajectory': True,
        'k': np.array([15]),
        'T0': np.array([0.5])
    }

    print('Create Environment.')
    env = gym.make('cryoenv:cryoenv-continuous-v0',
                   **env_kwargs,
                   )

    # print('Check Environment.')
    # check_env(env)

    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))

    # ------------------------------------------------
    # DEFINE AGENT PARAMETERS
    # ------------------------------------------------

    nmbr_agents = 2
    processes = 4
    multiprocessing = False
    train_episodes = 400
    max_steps = 10000
    test_steps = 100

    alpha = 0.7  # learning rate
    discount_factor = 0.65  # gamma
    if not isinstance(alpha, list):
        alpha = alpha * np.ones(nmbr_agents)
    if not isinstance(discount_factor, list):
        discount_factor = discount_factor * np.ones(nmbr_agents)
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 1 / train_episodes

    training = True
    testing = True
    save_q_after_ep = [200, train_episodes - 1]
    save_q_for_agent = [0, 1]

    # Creating lists to keep track of reward and epsilon values
    training_rewards = np.empty((nmbr_agents, train_episodes), dtype=float)
    epsilons = np.empty((nmbr_agents, train_episodes), dtype=float)

    # training loop
    Q = np.empty((env.observation_space.n, env.action_space.n), dtype=np.float32)
    kwargs = dict((name, eval(name)) for name in ['Q',
                                                  'env',
                                                  'training_rewards',
                                                  'epsilons',
                                                  'train_episodes',
                                                  'max_steps',
                                                  'epsilon',
                                                  'min_epsilon',
                                                  'max_epsilon',
                                                  'decay',
                                                  'multiprocessing',
                                                  'save_q_after_ep',
                                                  'save_q_for_agent'])

    if training:
        # ------------------------------------------------
        # TRAINING
        # ------------------------------------------------

        print('Training...')
        if multiprocessing:
            pool = mp.Pool(processes=processes)
            train_wrapper = partial(partial(train_agent, **kwargs))
            pool.map(train_wrapper, range(nmbr_agents))
        else:
            for agent in range(nmbr_agents):  # pool
                print('\nTrain Agent Nmbr: ', agent)
                Q[:] = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
                # print('Shape Q table: ', kwargs['Q'].shape)
                train_agent(agent,
                            alpha=alpha[agent],
                            discount_factor=discount_factor[agent],
                            **kwargs)

        # ------------------------------------------------
        # TRAINING PLOT
        # ------------------------------------------------

        print('Plots...')
        fig, host = plt.subplots(figsize=(8, 5))

        x = range(train_episodes)

        par1 = host.twinx()

        host.set_xlabel("Episode")
        host.set_ylabel("Average Reward")
        par1.set_ylabel("Epsilon")

        color1 = 'green'
        color2 = 'grey'

        p1, = host.plot(x, np.mean(training_rewards, axis=0), color=color1, label="Reward", linewidth=2, zorder=15)
        up = np.mean(training_rewards, axis=0) + np.std(training_rewards, axis=0)
        low = np.mean(training_rewards, axis=0) - np.std(training_rewards, axis=0)
        _ = host.fill_between(x, y1=up, y2=low, color=color1, zorder=5, alpha=0.3)
        _ = host.text(x=0.5, y=0.95, s=r'$\alpha$: {}'.format(alpha), transform=host.transAxes, ha='center')
        _ = host.text(x=0.5, y=0.9, s=r'$\gamma$: {}'.format(discount_factor), transform=host.transAxes, ha='center')
        p2, = par1.plot(x, np.mean(epsilons, axis=0), color=color2, label="Epsilon", linewidth=2.5, alpha=0.5,
                        linestyle='dashed', zorder=10)

        par1.spines['right'].set_position(('outward', 0))

        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())

        fig.tight_layout()
        fig.suptitle('CryoEnvDiscrete v0: Q-Learning Training')
        plt.savefig(fname='training.pdf')

        plt.show()

    if testing:
        returns = np.empty((len(save_q_for_agent), len(save_q_after_ep)), dtype=float)
        for i, ag in enumerate(save_q_for_agent):
            for j, ep in enumerate(save_q_after_ep):
                # ------------------------------------------------
                # TESTING
                # ------------------------------------------------

                Q = np.load("q_table_ep" + str(ep) + "_ag" + str(ag) + ".npy")

                print('Testing...')
                test_agent(env,
                           Q,
                           steps=test_steps,
                           )

                # ------------------------------------------------
                # TESTING PLOT TRAJECTORIES
                # ------------------------------------------------

                rewards, _, wait, _, V_set, _, _, _ = env.get_trajectory()
                env_steps = range(test_steps)

                # plot the trajectories

                fig, host = plt.subplots(figsize=(8, 5))
                par1 = host.twinx()
                par2 = host.twinx()

                host.set_xlabel("Environment Steps")
                host.set_ylabel("Voltage Setpoint (a.u.)")
                par1.set_ylabel("Reward")
                par2.set_ylabel("Waiting Time (s)")

                color1 = 'red'
                color2 = 'black'
                color3 = 'grey'

                p1, = host.plot(env_steps, V_set[:, 0], color=color2, label="Voltage Setpoint", linewidth=2,
                                zorder=15)
                p2, = par1.plot(env_steps, rewards, color=color1, label="Reward", linewidth=2,
                                zorder=10, alpha=0.6)
                p3, = par2.plot(env_steps, wait, color=color3, label="Waiting Time", linestyle='dashed',
                                linewidth=2.5)

                par2.spines['right'].set_position(('outward', 60))

                host.yaxis.label.set_color(p1.get_color())
                par1.yaxis.label.set_color(p2.get_color())
                par2.yaxis.label.set_color(p3.get_color())

                fig.tight_layout()

                plt.title('Trajectories Agent {} Episode {}'.format(ag, ep))

                plt.show()

                returns[i, j] = np.sum(rewards)

        # ------------------------------------------------
        # TESTING PLOT PARAMETERS
        # ------------------------------------------------

        plt.close()

        labels = [r'$\gamma$ ' + str(g) + r', $\alpha$ ' + str(a) for g, a in zip(discount_factor, alpha)]

        x = np.arange(len(labels))  # the label locations
        width = 0.7 / len(save_q_after_ep)  # the width of the bars

        fig, ax = plt.subplots()
        rects = []
        for i, (ret, ep) in enumerate(zip(returns.T, save_q_after_ep)):
            rects.append(ax.bar(x - 0.35 + i * width, ret, width, label='Episode {}'.format(ep)))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Return')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()

        plt.show()
