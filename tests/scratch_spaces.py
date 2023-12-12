import gymnasium as gym
import warnings
import numpy as np

warnings.simplefilter('ignore')
np.random.seed(0)
# gym.logger.set_level(40)

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

print(env.action_space)
print(env.observation_space)
print(type(env.action_space) == gym.spaces.box.Box)
print(type(env.observation_space) == gym.spaces.box.Box)
print(env.action_space.shape)
print(env.action_space.low)
print(env.action_space.high)