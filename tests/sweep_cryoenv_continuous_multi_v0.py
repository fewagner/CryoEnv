# imports
import gym
import ipdb
from ipdb.__main__ import set_trace
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.env_checker import check_env
from cryoenv.envs._cryoenv_discrete_v0 import action_to_discrete, \
    observation_from_discrete
from cryoenv.envs._cryoenv_continuous_v0 import CryoEnvContinuous_v0

gym.logger.set_level(40)

# constants
TEST_PULSE_AMPLITUDES = [50, 30, 5]
env_kwargs = {
    'dac_iv': (0, 99),
    'wait_iv': (2., 100.),
    'ph_iv': (0, 0.99),
    'heater_resistance': np.array([100., 100.]),
    'thermal_link_channels': np.array([[1, 0.1], [0.1, 1]]),
    'thermal_link_heatbath': np.array([1., 1.]),
    'temperature_heatbath': 0.,
    'min_ph': 0.01,
    'g': np.array([0.0001, 0.0001]),
    'T_hyst': np.array([0.1, 0.1]),
    'T_hyst_reset': np.array([0.9, 0.9]),
    'hyst_wait': np.array([50, 50]),
    'tpa': 50,
    'env_fluctuations': 0.005,
    'model_pileup_drops': True,
    'prob_drop': np.array([1e-3, 1e-3]),  # per second!
    'prob_pileup': np.array([0.1, 0.1]),
    'save_trajectory': True,
    'k': np.array([15, 15]),
    'T0': 0.5,
    'incentive_reset': 1e-2,
}


# main

dac_iv_steps = 200

nmbr_detectors = env_kwargs['k'].shape[0]
trajectories = []


grid = np.arange(-0.5, 1.5, 0.01)
set_point_pos = np.diff(
    CryoEnvContinuous_v0.sensor_model(
        CryoEnvContinuous_v0,
        grid,
        env_kwargs['k'],
        env_kwargs['T0']
    ),
    axis=1
).argmax(axis=1)
set_point_pos = set_point_pos - 20  # arbitrary offset


for di in range(nmbr_detectors):
    for amp in TEST_PULSE_AMPLITUDES:
        env_kwargs['tpa'] = amp
        env = gym.make('cryoenv:cryoenv-continuous-v0',
                       **env_kwargs,
                       )

        obs = env.reset()
        print("Exemplary State {}".format(obs))
        print("Exemplary State denormed {}".format(env.denorm_state(obs)))

        # print('Check Environment.')
        check_env(env)

        print('Sweep from top to bottom.')
        count = 0
        env.reset()

        for v_s in np.linspace(env_kwargs['dac_iv'][1],
                               env_kwargs['dac_iv'][0],
                               dac_iv_steps):
            dacs_i = set_point_pos / grid.shape[0] * \
                (env_kwargs['dac_iv'][1] - env_kwargs['dac_iv'][0])
            dacs_i[di] = v_s

            # import ipdb; ipdb.set_trace()
            action = np.array([dacs_i, [10] * nmbr_detectors]
                              ).flatten('F')  # wait = 10
            # print(f'action: {action}')
            new_state, _, _, _ = env._step(action=action)
            count += 1
        # import ipdb; ipdb.set_trace()
        trajectories.append(env.get_trajectory())


print('Plot the sensor model.')
k, T0 = env.k, env.T0
print('k, T0: ', k, T0)
plt.close()
grid = np.array([np.arange(-0.5, 1.5, 0.01)] * nmbr_detectors).T
smodel = env.sensor_model(grid, k, T0)

if False:
    for i in range(nmbr_detectors):
        plt.figure(i)
        plt.subplot(211 + 10 * i)
        plt.plot(grid[:, i], smodel[i], color='C0', linewidth=3)

        plt.axvline(x=0, color='black')
        plt.axvline(x=1, color='black')
        plt.xlabel('Simplified Temperature (a.u.)')
        plt.ylabel('Simplified Resistance (a.u.)')

    plt.title('Sensor model')
    plt.tight_layout()
    plt.savefig(fname='results/sensor.pdf')
    plt.show()
elif False:
    fig, host = plt.subplots(nrows=1, ncols=nmbr_detectors, figsize=(8, 4))
    for i in range(nmbr_detectors):
        host[i].plot(grid[:, i], smodel[i], color='C0', linewidth=3)

        host[i].axvline(x=0, color='black')
        host[i].axvline(x=1, color='black')
        host[i].set_xlabel('Simplified Temperature (a.u.)')
        host[i].set_ylabel('Simplified Resistance (a.u.)')
        host[i].set_title('Sensor model - Ch. {}'.format(i))
    plt.tight_layout()
    plt.savefig(fname='results/sensor_models.pdf')
    plt.show()


####
# we want to see the trajectories
print('Plot all trajectories.')


for i in range(nmbr_detectors):
    # low pulses
    rewards, dV, wait, reset, new_dac, new_ph, T, T_inj = trajectories[0 + 3 * i]
    _, _, _, _, _, new_ph_mid, _, _ = trajectories[1 + 3 * i]
    _, _, _, _, _, new_ph_small, _, _ = trajectories[2 + 3 * i]

    fig, host = plt.subplots(nrows=1, ncols=nmbr_detectors, figsize=(10, 4))

    par1, par2 = [[None] * nmbr_detectors] * nmbr_detectors
    par1[0] = host[0].twinx()
    par2[0] = host[0].twinx()
    par1[1] = host[1].twinx()
    par2[1] = host[1].twinx()

    # here set limits
    for j, h, p1, p2 in zip(range(len(host)), host, par1, par2):
        h.set_xlabel("Voltage Setpoint (a.u.)")
        h.set_ylabel("Reward")
        p1.set_ylabel("Control/Test Pulse Height (a.u.)")
        p2.set_ylabel("Simplified Temperature (a.u.)")

        color1 = 'red'
        color2 = 'black'
        color3 = 'grey'
        # ipdb.set_trace()
        p1p, =  h.plot(new_dac[:, i], rewards, color=color1,
                       label="Reward", linewidth=1, zorder=10)
        p2p, = p1.plot(new_dac[:, i], new_ph[:, j], color=color2,
                       label="Control Pulse Height", linewidth=1, marker='.', linestyle='', zorder=15)
        _ = p1.plot(new_dac[:, i], new_ph_mid[:, j], color=color2,
                    linewidth=1, marker='.', linestyle='', zorder=15)
        _ = p1.plot(new_dac[:, i], new_ph_small[:, j], color=color2,
                    linewidth=1, marker='.', linestyle='', zorder=15)
        p3p, = p2.plot(new_dac[:, i], T[:, j], color=color3,
                       label="Simplified Temperature", linewidth=1, linestyle='dashed')
        temp = env.sensor_model(T, k, T0)
        _ = p2.plot(new_dac[:, i], temp[j], color='C0', linewidth=3, alpha=0.5)
        # _    = p2.plot(new_dac[:,i], env.sensor_model(T, k, T0)[:,i], color='C0', linewidth=3, alpha=0.5)

        # lns = [p1, p2, p3]
        # host.legend(handles=lns, loc='best')

        # right, left, top, bottom
        p2.spines['right'].set_position(('outward', 60))

        # no x-ticks
        # par2.xaxis.set_ticks([])

        h.yaxis.label.set_color(p1p.get_color())
        p1.yaxis.label.set_color(p2p.get_color())
        p2.yaxis.label.set_color(p3p.get_color())

    fig.tight_layout()
    plt.tight_layout()

    plt.savefig(fname='results/sweep_cont_{}.pdf'.format(i))

    plt.show()
