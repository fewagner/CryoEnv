# imports
import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.env_checker import check_env
from cryoenv.envs._cryoenv_discrete_v0 import action_to_discrete, observation_from_discrete

gym.logger.set_level(40)

# constants
V_SET_IV = (0, 99)
V_SET_STEP = 1
PH_IV = (0, 0.99995)
PH_STEP = 0.00005
WAIT_IV = (2, 100)
WAIT_STEP = 2
HEATER_RESISTANCE = np.array([100.])
THERMAL_LINK_CHANNELS = np.array([[1.]])
THERMAL_LINK_HEATBATH = np.array([1.])
TEMPERATURE_HEATBATH = 0.
MIN_PH = 0.1
G = np.array([0.001])
T_HYST = np.array([0.001])
CONTROL_PULSE_AMPLITUDES = [50, 20, 5]
ENV_FLUCTUATIONS = 0.001
k = np.array([15])
T0 = np.array([0.5])

# main

trajectories = []

for amp in CONTROL_PULSE_AMPLITUDES:

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
                   control_pulse_amplitude=amp,
                   env_fluctuations=ENV_FLUCTUATIONS,
                   save_trajectory=True,
                   k=k,
                   T0=T0,
                   )

    # print('Check Environment.')
    # check_env(env)

    print('Sweep from top to bottom.')
    count = 0
    env.reset()
    for v_s in np.arange(V_SET_IV[1], V_SET_IV[0] - V_SET_STEP, - V_SET_STEP):
        # print(f'V_set: {v_s}')
        action = env.action_to_discrete(reset=np.zeros([1], dtype=bool),
                                    V_decrease=V_SET_STEP * np.ones([1], dtype=float),
                                    wait=np.array([10], dtype=float))
        new_state, _, _, _ = env.step(action=action)
        new_V_set, new_ph = env.observation_from_discrete(new_state)
        count += 1

    trajectories.append(env.get_trajectory())

print('Plot the sensor model.')
k, T0 = env.k, env.T0
print('k, T0: ', k, T0)
plt.close()
grid = np.arange(-0.5, 1.5, 0.01)
plt.plot(grid, env.sensor_model(grid, k, T0), color='C0', linewidth=3)
plt.axvline(x=0, color='black')
plt.axvline(x=1, color='black')
plt.title('Sensor model')
plt.xlabel('Simplified Temperature (a.u.)')
plt.ylabel('Simplified Resistance (a.u.)')
plt.savefig(fname='sensor.pdf')
plt.show()


####
# we want to see the trajectories
print('Plot all trajectories.')

rewards, dV, wait, reset, new_V_set, new_ph, T, T_inj = trajectories[0]
_, _, _, _, _, new_ph_mid, _, _ = trajectories[1]
_, _, _, _, _, new_ph_small, _, _ = trajectories[2]

fig, host = plt.subplots(figsize=(8, 5))

par1 = host.twinx()
par2 = host.twinx()

# here set limits

host.set_xlabel("Voltage Setpoint (a.u.)")
host.set_ylabel("Reward")
par1.set_ylabel("Control/Test Pulse Height (a.u.)")
par2.set_ylabel("Simplified Temperature (a.u.)")

color1 = 'red'
color2 = 'black'
color3 = 'grey'

p1, = host.plot(new_V_set, rewards, color=color1, label="Reward", linewidth=2, zorder=10)
p2, = par1.plot(new_V_set, new_ph, color=color2, label="Control Pulse Height", marker='.', linestyle='', zorder=15)
_ = par1.plot(new_V_set, new_ph_mid, color=color2, marker='.', linestyle='', zorder=15)
_ = par1.plot(new_V_set, new_ph_small, color=color2, marker='.', linestyle='', zorder=15)
p3, = par2.plot(new_V_set, T, color=color3, label="Simplified Temperature", linestyle='dashed', linewidth=2.5)
_ = par2.plot(new_V_set, env.sensor_model(T, k, T0), color='C0', linewidth=3, alpha=0.5)

# lns = [p1, p2, p3]
# host.legend(handles=lns, loc='best')

# right, left, top, bottom
par2.spines['right'].set_position(('outward', 60))

# no x-ticks
# par2.xaxis.set_ticks([])

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

fig.tight_layout()

plt.savefig(fname='sweep.pdf')

plt.show()
