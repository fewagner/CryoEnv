import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import collections
import math
import matplotlib.pyplot as plt
from ..cryosig._detector_model import DetectorModule
from ..cryosig._parameter_sampler import sample_parameters


class CryoEnvSigWrapper(gym.Env):
    """
    TODO
    """

    metadata = {'render_modes': ['human', 'mpl']}

    def __init__(self, pars=None, omega=1e-2, sample_pars=False, render_mode=None,
                 rand_start=False, relax_time=90, log_reward=False, tpa_in_state=True,
                 ):
        if pars is not None:
            self.pars = pars
        else:
            self.pars = {}
        if sample_pars:
            self.pars = sample_parameters(**self.pars)
        self.detector = DetectorModule(**self.pars)
        self.nmbr_actions = self.detector.nmbr_heater + self.detector.nmbr_tes
        if tpa_in_state:
            self.nmbr_observations = 3 * self.detector.nmbr_tes + 3 * self.detector.nmbr_heater
        else:
            self.nmbr_observations = 3 * self.detector.nmbr_tes + 2 * self.detector.nmbr_heater
        self.ntes = self.detector.nmbr_tes
        self.nheater = self.detector.nmbr_heater
        self.omega = omega
        self.rand_start = rand_start
        self.relax_time = relax_time
        self.log_reward = log_reward
        self.tpa_in_state = tpa_in_state
        # self.dac_memory = np.zeros(self.detector.nmbr_heater)
        # self.Ib_memory = np.zeros(self.detector.nmbr_tes)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=- np.ones(self.nmbr_actions),
                                       high=np.ones(self.nmbr_actions),
                                       dtype=np.float32)  # DACs, IBs
        self.observation_space = spaces.Box(low=- np.ones(self.nmbr_observations),
                                            high=np.ones(self.nmbr_observations),
                                            dtype=np.float32)  # PHs, RMSs, IBs, DACs, TPA, CPs

        _ = self.reset()

    # public

    def step(self, action):

        info = {}

        self.detector.set_control(dac=action[:self.nheater],
                                  Ib=action[self.nheater:self.nheater + self.ntes],
                                  norm=True)
        self.detector.wait(seconds=self.detector.tp_interval - self.detector.t[-1])
        self.detector.trigger(er=np.zeros(self.detector.nmbr_components),
                              tpa=self.detector.tpa_queue[self.detector.tpa_idx],
                              verb=False)
        self.detector.tpa_idx += 1
        if self.detector.tpa_idx + 1 > len(self.detector.tpa_queue):
            self.detector.tpa_idx = 0

        new_state = np.ones(self.nmbr_observations)
        new_state[:self.ntes] = self.detector.get('ph', norm=True)
        new_state[self.ntes:2 * self.ntes] = self.detector.get('rms', norm=True)
        new_state[2 * self.ntes:3 * self.ntes] = self.detector.get('Ib', norm=True)
        new_state[3 * self.ntes:3 * self.ntes + self.nheater] = self.detector.get('dac', norm=True)
        if self.tpa_in_state:
            # relax_factor = np.exp(-self.detector.tp_interval / self.relax_time)
            new_state[3 * self.ntes + self.nheater:3 * self.ntes + 2 * self.nheater] = self.detector.get('tpa',
                                                                                                         norm=True)
            # new_state[3 * self.ntes + 2 * self.nheater:4 * self.ntes + 2 * self.nheater] = \
            #     self.state[3 * self.ntes + 2 * self.nheater:4 * self.ntes + 2 * self.nheater] * relax_factor - \
            #     (1 - relax_factor) * self.detector.get('Ib', norm=True)
            # new_state[4 * self.ntes + 2 * self.nheater:4 * self.ntes + 3 * self.nheater] = \
            #     self.state[4 * self.ntes + 2 * self.nheater:4 * self.ntes + 3 * self.nheater] * relax_factor - \
            #     (1 - relax_factor) * self.detector.get('dac', norm=True)

        if not self.log_reward:
            reward = - np.sum(
                self.detector.rms * self.detector.tpa / np.maximum(self.detector.ph, self.detector.rms))
        else:
            reward = - np.log(
                np.sum(self.detector.rms * self.detector.tpa / np.maximum(self.detector.ph, self.detector.rms)))
        reward -= self.omega * np.sum((new_state[self.ntes:] - self.state[self.ntes:]) ** 2)

        self.state = new_state

        terminated = False
        truncated = False

        return new_state, reward, terminated, truncated, info

    def reset(self):

        info = {}

        if not self.rand_start:
            self.detector.clear_buffer()
            self.detector.set_control(dac=-np.ones(self.detector.nmbr_heater),
                                      Ib=-np.ones(self.detector.nmbr_tes),
                                      norm=True)
            self.state = - np.ones(self.nmbr_observations)

        else:
            info = {}

            self.detector.clear_buffer()
            action = self.action_space.sample()
            randflag = np.random.choice([True, False], size=self.nmbr_actions)
            action[randflag] = np.random.choice([-1, 1], size=np.sum(randflag))
            self.detector.tpa_idx = np.random.choice(len(self.detector.tpa_queue))

            self.detector.set_control(dac=action[:self.nheater],
                                      Ib=action[self.nheater:self.nheater + self.ntes],
                                      norm=True)
            self.detector.wait(seconds=2 * self.relax_time)
            self.detector.trigger(er=np.zeros(self.detector.nmbr_components),
                                  tpa=self.detector.tpa_queue[self.detector.tpa_idx],
                                  verb=False)
            self.detector.tpa_idx += 1
            if self.detector.tpa_idx + 1 > len(self.detector.tpa_queue):
                self.detector.tpa_idx = 0

            new_state = np.ones(self.nmbr_observations)
            new_state[:self.ntes] = self.detector.get('ph', norm=True)
            new_state[self.ntes:2 * self.ntes] = self.detector.get('rms', norm=True)
            new_state[2 * self.ntes:3 * self.ntes] = self.detector.get('Ib', norm=True)
            new_state[3 * self.ntes:3 * self.ntes + self.nheater] = self.detector.get('dac', norm=True)
            if self.tpa_in_state:
                new_state[3 * self.ntes + self.nheater:3 * self.ntes + 2 * self.nheater] = self.detector.get('tpa',
                                                                                                             norm=True)
                # new_state[3 * self.ntes + 2 * self.nheater:4 * self.ntes + 2 * self.nheater] = self.detector.get('Ib',
                #                                                                                                  norm=True)
                # new_state[4 * self.ntes + 2 * self.nheater:4 * self.ntes + 3 * self.nheater] = self.detector.get('dac',
                #                                                                                                  norm=True)

            self.state = new_state

        return self.state, info

    def render(self, save_path=None):
        if self.render_mode == "human":
            # self.detector.plot_event(show=True)
            self.detector.plot_temperatures(show=True)
            self.detector.plot_tes(show=True)

        elif self.render_mode == "mpl":
            # self.detector.plot_event(show=False)
            self.detector.plot_temperatures(show=False)
            if save_path is not None:
                plt.savefig(save_path + '_temps')
            plt.close()
            self.detector.plot_tes(show=False)
            if save_path is not None:
                plt.savefig(save_path + '_tes')
            plt.close()

    def close(self):
        _ = self.reset()
