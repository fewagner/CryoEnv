# imports
import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.env_checker import check_env

gym.logger.set_level(40)

print('Create Environment.')
env = gym.make('cryoenv:cryoenv-discrete-v0',
               V_set_iv=(0, 99),
               V_set_step=1,
               ph_iv=(0, 0.99),
               ph_step=0.01,
               wait_iv=(2, 100),
               wait_step=2,
               heater_resistance=np.array([100.]),
               thermal_link_channels=np.array([[1.]]),
               thermal_link_heatbath=np.array([1.]),
               temperature_heatbath=0.,
               min_ph=0.2,
               g=np.array([0.001]),
               T_hyst=np.array([0.001]),
               control_pulse_amplitude=10,
               env_fluctuations=1,
               save_trajectory=False,
               )

print('Check Environment.')
check_env(env)


