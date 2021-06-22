# imports
import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.env_checker import check_env
from cryoenv.envs._cryoenv_discrete_v0 import action_to_discrete, observation_from_discrete

gym.logger.set_level(40)

# constants
CONTROL_PULSE_AMPLITUDES = [50, 20, 5]
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
    'T0': np.array([0.5]),
}

# main

trajectories = []

for amp in CONTROL_PULSE_AMPLITUDES:
    env_kwargs['control_pulse_amplitude'] = amp

    print('Create Environment.')
    env = gym.make('cryoenv:cryoenv-continuous-v0',
                   **env_kwargs,
                   )

    obs = env.reset()
    print("Exemplary State {}".format(obs))

    # print('Check Environment.')
    check_env(env)

    print('Sweep from top to bottom.')
    count = 0
    env.reset()
    for v_s in np.arange(env_kwargs['V_set_iv'][1],
                         env_kwargs['V_set_iv'][0] - env_kwargs['V_set_step'],
                         - env_kwargs['V_set_step']):
        action = np.array([[v_s, ], [10, ]]).T  # wait = 10
        # print(f'action: {action}')
        new_state, _, _, _ = env.step(action=action)
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
