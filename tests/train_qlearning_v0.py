
import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_checker import check_env
from cryoenv.agents import Interpolator, EpsilonGreedy, QLearning

import warnings

warnings.simplefilter('ignore')
np.random.seed(0)
gym.logger.set_level(40)


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
        'min_ph': 0.01,
        'g': np.array([0.0001]),
        'T_hyst': np.array([0.1]),
        'T_hyst_reset': np.array([0.9]),
        'hyst_wait': np.array([50]),
        'control_pulse_amplitude': 50,
        'env_fluctuations': 0.005,
        'model_pileup_drops': True,
        'prob_drop': np.array([1e-3]),  # per second!
        'prob_pileup': np.array([0.1]),
        'save_trajectory': True,
        'k': np.array([15]),
        'T0': np.array([0.5]),
        'incentive_reset': 1e-2,
    }

    print('Create Environment.')
    env = gym.make('cryoenv:cryoenv-continuous-v0',
                   **env_kwargs,
                   )

    check_env(env)

    # ------------------------------------------------
    # DEFINE AGENT PARAMETERS
    # ------------------------------------------------

    nmbr_agents = 1
    train_steps = 10000
    test_steps = 100
    smoothing = int(train_steps/500)
    assert train_steps % smoothing == 0, 'smoothing must be divisor of train_steps!'
    plot_axis = int(train_steps / smoothing)
    training = True
    # testing = True
    show = True

    # Creating lists to keep track of reward and epsilon values
    training_rewards = np.empty((nmbr_agents, plot_axis), dtype=float)

    if training:
        # ------------------------------------------------
        # TRAINING
        # ------------------------------------------------

        print('Training...')
        for agi in range(nmbr_agents):
            print('Learn Agent {}:'.format(agi))

            value_function = Interpolator(maxlen=500, initval=0, method='nearest')
            policy = EpsilonGreedy(epsilon=1)
            agent = QLearning(env=env, policy=policy, value_function=value_function)

            agent.learn(nmbr_steps=train_steps, learning_rate=0.5, discount_factor=0.6, max_epsilon=1, min_epsilon=0)
            if agi == 0:
                agent.save("model_continuous")

            rew, _, _, _, _, _, _, _, = env.get_trajectory()
            training_rewards[agi, :] = np.mean(rew.reshape(-1, smoothing), axis=1)
            env.reset()
            del agent

    # ------------------------------------------------
    # TRAINING PLOT
    # ------------------------------------------------

        print('Plots...')
        fig, host = plt.subplots(figsize=(8, 5))

        x = np.arange(plot_axis)*smoothing

        host.set_xlabel("Environment Steps")
        host.set_ylabel("Average Reward")

        color1 = 'green'

        p1, = host.plot(x, np.mean(training_rewards, axis=0), color=color1, label="Reward", linewidth=2, zorder=15)
        up = np.mean(training_rewards, axis=0) + np.std(training_rewards, axis=0)
        low = np.mean(training_rewards, axis=0) - np.std(training_rewards, axis=0)
        _ = host.fill_between(x, y1=up, y2=low, color=color1, zorder=5, alpha=0.3)

        host.yaxis.label.set_color(p1.get_color())

        fig.tight_layout()
        fig.suptitle('CryoEnvContinuous v0: Training')
        plt.savefig(fname='results/training_cont.pdf')

        if show:
            plt.show()

    # if testing:
    #     # ------------------------------------------------
    #     # TESTING
    #     # ------------------------------------------------
    #
    # print('Testing...')
    # obs = env.reset()
    # for i in range(100):
    #     action = agent.predict(obs)
    #     nobs, rewards, dones, info = env.step(action)
    #     print(f'S {obs} --> A {action} --> R {rewards} S {nobs}')
    #     obs = nobs