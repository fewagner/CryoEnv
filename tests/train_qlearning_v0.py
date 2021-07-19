
import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_checker import check_env
import cryoenv

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

    value_function = cryoenv.agents.Interpolator(maxlen=100, initval=0, method='nearest')
    policy = cryoenv.agents.EpsilonGreedy(epsilon=1)
    agent = cryoenv.agents.QLearning(env, policy, value_function)

    agent.learn(nmbr_steps=100, learning_rate=0.1, discount_factor=0.6, max_epsilon=1, min_epsilon=0)

    print('Testing...')
    obs = env.reset()
    for i in range(100):
        action = agent.predict(obs)
        nobs, rewards, dones, info = env.step(action)
        print(f'S {obs} --> A {action} --> R {rewards} S {nobs}')
        obs = nobs