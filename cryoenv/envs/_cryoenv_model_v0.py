import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import collections
import math


def linear_map(vals, k, d):
    return k * vals + d

def inv_linear_map(vals, k, d):
    return (vals - d) / k

class CryoEnvModel_v0(gym.Env):
    """
    TODO

    state: (repeat for each channel)
        - 0 : ph
        - 1 : rms
        - 2 : dac
        - 3 : bias_current
        - 4 : tpa

    action:
        - 0 : ramping_speed
        - 1 : bias_current
        - 2 : tpa

    reward:
        -

    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 detectors: object = None,
                 frames: int = 1,
                 trajectory: list = None,
                 ):
        self.detectors = detectors
        self.frames = frames
        self.nmbr_channels = detectors.nmbr_channels
        self.nmbr_actions = 3 * self.nmbr_channels  # will change later
        self.nmbr_observations = self.frames * self.nmbr_channels  # will change later

        self.action_space = spaces.Box(low=- np.ones(self.nmbr_actions * self.nmbr_channels),
                                       high=np.ones(self.nmbr_actions * self.nmbr_channels),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=- np.ones(self.nmbr_observations * self.nmbr_channels),
                                            high=np.ones(self.nmbr_observations * self.nmbr_channels),
                                            dtype=np.float32)

        self.trajectory = trajectory

    # public

    def load_detectors(self, detectors):
        self.detectors = detectors

    def step(self, action):
        denormed_action = self.denorm_action(action)
        new_state, reward, done, info = self._step(denormed_action)
        normed_new_state = self.norm_state(new_state)
        return normed_new_state, reward, done, info

    def reset(self):
        pass  # TODO

        self.reset_trajectories()
        return self.norm_state(self.state)

    def get_trajectory(self):
        pass  # TODO

    def reset_trajectories(self):
        pass  # TODO

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    # private

    def _step(self, action):
        # unpack action
        dac, Ib, tpa = action.reshape((self.nmbr_channels, 3)).T

        # unpack current state
        ph, rms, dac, Ib, tpa = self.state.reshape((self.nmbr_channels, 5)).T

        # get the next V sets and phs

        pass  # TODO

        new_ph, new_rms = ..., ...

        # pack new_state
        new_state = np.array([new_ph, new_rms, dac, Ib, tpa]).T.reshape(-1)

        # get the reward
        reward = self._reward(new_state, action)

        # update state
        self.state = new_state

        # save trajectory
        if self.trajectory:
            self.trajectory.append((self.state, action, reward, new_state))

        # the task is continuing
        done = False

        info = {}

        return new_state, reward, done, info

    def _reward(self, state, action):

        reward = ...  # TODO

        return reward

    def _norm_action(self, action):
        """
        action:
        - 0 : ramping_speed
        - 1 : bias_current
        - 2 : tpa
        """
        normed_action = np.copy(action)

        normed_action[0::3] = linear_map(normed_action[0::3], *self.detectors.pars['V_conversion'])
        normed_action[1::3] = linear_map(normed_action[1::3], *self.detectors.pars['Ib_conversion'])
        normed_action[2::3] = linear_map(normed_action[2::3], *self.detectors.pars['tpa_conversion'])

        return normed_action

    def _denorm_action(self, normed_action):
        """
        action:
        - 0 : ramping_speed
        - 1 : bias_current
        - 2 : tpa
        """
        action = np.copy(normed_action)

        action[0::3] = inv_linear_map(action[0::3], *self.detectors.pars['V_conversion'])
        action[1::3] = inv_linear_map(action[1::3], *self.detectors.pars['Ib_conversion'])
        action[2::3] = inv_linear_map(action[2::3], *self.detectors.pars['tpa_conversion'])

        return action

    def _norm_state(self, state):
        """
        state: (repeat for each channel)
        - 0 : ph
        - 1 : rms
        - 2 : dac
        - 3 : bias_current
        - 4 : tpa
        """
        normed_state = np.copy(state)

        normed_state[0::3] = linear_map(normed_state[0::3], *self.detectors.pars['V_conversion'])
        normed_state[1::3] = linear_map(normed_state[1::3], *self.detectors.pars['V_conversion'])
        normed_state[2::3] = linear_map(normed_state[2::3], *self.detectors.pars['V_conversion'])
        normed_state[3::3] = linear_map(normed_state[3::3], *self.detectors.pars['Ib_conversion'])
        normed_state[4::3] = linear_map(normed_state[4::3], *self.detectors.pars['tpa_conversion'])

        return normed_state

    def _denorm_state(self, normed_state):
        """
        state: (repeat for each channel)
        - 0 : ph
        - 1 : rms
        - 2 : dac
        - 3 : bias_current
        - 4 : tpa
        """
        state = np.copy(normed_state)

        state[0::3] = inv_linear_map(state[0::3], *self.detectors.pars['V_conversion'])
        state[1::3] = inv_linear_map(state[1::3], *self.detectors.pars['V_conversion'])
        state[2::3] = inv_linear_map(state[2::3], *self.detectors.pars['V_conversion'])
        state[3::3] = inv_linear_map(state[3::3], *self.detectors.pars['Ib_conversion'])
        state[4::3] = inv_linear_map(state[4::3], *self.detectors.pars['tpa_conversion'])

        return state
