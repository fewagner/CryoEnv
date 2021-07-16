import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, SAC
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback

import warnings

warnings.simplefilter('ignore')
np.random.seed(0)
gym.logger.set_level(40)


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


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

    # print('Check Environment.')
    check_env(env)

    # ------------------------------------------------
    # DEFINE AGENT PARAMETERS
    # ------------------------------------------------

    nmbr_agents = 1
    train_steps = 100000
    test_steps = 100
    smoothing = int(train_steps/500)
    assert train_steps % smoothing == 0, 'smoothing must be divisor of train_steps!'
    plot_axis = int(train_steps / smoothing)
    training = True
    testing = True
    show = True

    # Creating lists to keep track of reward and epsilon values
    training_rewards = np.empty((nmbr_agents, plot_axis), dtype=float)

    if training:
        # ------------------------------------------------
        # TRAINING
        # ------------------------------------------------

        print('Training...')
        for agent in range(nmbr_agents):
            print('Learn Agent {}:'.format(agent))
            model = A2C("MlpPolicy", env, verbose=False, gamma=0.6)
            with ProgressBarManager(train_steps) as callback:
                model.learn(total_timesteps=train_steps, callback=callback)
            if agent == 0:
                model.save("model_continuous")
            rew, _, _, _, _, _, _, _, = env.get_trajectory()
            training_rewards[agent, :] = np.mean(rew.reshape(-1, smoothing), axis=1)
            env.reset()
            del model

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

    if testing:
        # ------------------------------------------------
        # TESTING
        # ------------------------------------------------

        # Q = np.load("q_table_ep" + str(ep) + "_ag" + str(ag) + ".npy")

        print('Testing...')
        model = A2C.load("model_continuous")
        obs = env.reset()
        for i in range(test_steps):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)

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

        plt.title('CryoEnvContinuous v0: Test Trajectories')

        if show:
            plt.show()
