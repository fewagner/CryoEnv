# imports
import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.env_checker import check_env
from cryoenv.envs._cryoenv_discrete_v0 import action_to_discrete

gym.logger.set_level(40)

# constants
V_SET_IV = (0, 99)
V_SET_STEP = 1
PH_IV = (0, 0.99)
PH_STEP = 0.01
WAIT_IV = (2, 100)
WAIT_STEP = 2
HEATER_RESISTANCE = np.array([100.])
THERMAL_LINK_CHANNELS = np.array([[1.]])
THERMAL_LINK_HEATBATH = np.array([1.])
TEMPERATURE_HEATBATH = 0.
MIN_PH = 0.2
G = np.array([0.001])
T_HYST = np.array([0.001])
CONTROL_PULSE_AMPLITUDE = 10
ENV_FLUCTUATIONS = 1

# main

print('Create Environment.')
env = gym.make('cryoenv:cryoenv-discrete-v0',
               V_set_iv=V_SET_IV,
               V_set_step=V_SET_STEP,
               ph_iv=PH_IV,
               ph_step=PH_STEP,
               wait_iv=WAIT_IV,
               wait_step=WAIT_STEP,
               heater_resistance=HEATER_RESISTANCE,
               thermal_link_channels=THERMAL_LINK_CHANNELS,
               thermal_link_heatbath=THERMAL_LINK_HEATBATH,
               temperature_heatbath=TEMPERATURE_HEATBATH,
               min_ph=MIN_PH,
               g=G,
               T_hyst=T_HYST,
               control_pulse_amplitude=CONTROL_PULSE_AMPLITUDE,
               env_fluctuations=ENV_FLUCTUATIONS,
               save_trajectory=True,
               )

print('Check Environment.')
check_env(env)

print('Sweep from top to bottom.')
for v_s in range(V_SET_IV[1], V_SET_IV[0], - V_SET_STEP):
    action = action_to_discrete(reset=np.zeros([1], dtype=bool),
                                V_decrease=V_SET_STEP*np.ones([1], dtype=float),
                                wait=5*np.ones([1], dtype=float))
    _ = env.step(action=action)

print('Plot the sensor model.')
k, T0 = env.k, env.T0
print('k, T0: ', k, T0)
plt.close()
grid = np.arange(-0.5, 1.5, 0.01)
plt.plot(grid, env.sensor_model(grid, k, T0))
plt.axvline(x=0, color='black')
plt.axvline(x=1, color='black')
plt.show()

# we want to see the trajectories
print('Plot all trajectories.')
vals = env.get_trajectory()
titles = ['rewards', 'V_decrease', 'wait', 'reset', 'new_V_set', 'new_ph', 'T', 'T_inj']

for arr, tit in zip(vals, titles):
    plt.close()
    plt.plot(arr)
    plt.title(tit)
    plt.ylabel('Value')
    plt.xlabel('Environment Step')
    plt.show()