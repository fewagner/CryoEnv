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

    metadata = {'render.modes': ['human']}

    def __init__(self, pars=None, omega=1e-2, sample_pars=False,
                 ):
        if pars is not None:
            self.pars = pars
        else:
            self.pars = {}
        if sample_pars:
            self.pars = sample_parameters(**self.pars)
        self.detector = DetectorModule(**self.pars)
        self.nmbr_actions = self.detector.nmbr_heater + self.detector.nmbr_tes
        self.nmbr_observations = 2 * self.detector.nmbr_tes + self.detector.nmbr_heater
        self.ntes = self.detector.nmbr_tes
        self.nheater = self.detector.nmbr_heater
        self.omega = omega

        self.action_space = spaces.Box(low=- np.ones(self.nmbr_actions),
                                       high=np.ones(self.nmbr_actions),
                                       dtype=np.float32)  # DACs, IBs
        self.observation_space = spaces.Box(low=- np.ones(self.nmbr_observations),
                                            high=np.ones(self.nmbr_observations),
                                            dtype=np.float32)  # PHs, DACs, IBs

        _ = self.reset()

    # public

    def step(self, action):

        info = {}

        self.detector.set_control(dac=action[:self.nheater],
                                  Ib=action[self.nheater:self.nheater + self.ntes],
                                  norm=True)
        self.detector.wait(seconds=self.detector.tp_interval - self.detector.t[-1])
        self.detector.trigger(er=0.,
                              tpa=self.detector.tpa_queue[self.detector.tpa_idx],
                              verb=False)
        self.detector.tpa_idx += 1
        if self.detector.tpa_idx + 1 > len(self.detector.tpa_queue):
            self.detector.tpa_idx = 0

        new_state = np.ones(self.nmbr_observations)
        new_state[:self.ntes] = self.detector.get('ph', norm=True)
        new_state[self.ntes:self.ntes + self.nheater] = self.detector.get('dac', norm=True)
        new_state[self.ntes + self.nheater:2 * self.ntes + self.nheater] = self.detector.get('Ib', norm=True)

        reward = - np.sum(self.detector.rms * self.detector.tpa / self.detector.ph) - \
                 self.omega * np.sum((new_state[self.ntes:] - self.state[self.ntes:]) ** 2)

        self.state = new_state

        done = False

        return new_state, reward, done, info

    def reset(self):
        self.detector.clear_buffer()
        self.detector.set_control(dac=-np.ones(self.detector.nmbr_heater),
                                  Ib=-np.ones(self.detector.nmbr_tes),
                                  norm=True)
        self.state = - np.ones(self.nmbr_observations)
        return self.state

    def render(self, mode='human', save_path=None):
        assert mode in ["human", "mpl"], "Invalid mode, must be either \"human\" or \"mpl\""
        if mode == "human":
            self.detector.plot_event(show=True)

        elif mode == "mpl":
            self.detector.plot_event(show=False)
            if save_path is not None:
                plt.savefig(save_path)
            plt.close()

    def close(self):
        _ = self.reset()