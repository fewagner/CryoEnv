
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class CryoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 action_low=np.array([[0, 1, 0]]),  # first action is V_set, second is waiting time, third is reset prob
                 action_high=np.array([[1, 100, 1]]),
                 oberservation_low=np.array([[0, 0]]),  # first observation is V_set, second is PH
                 oberservation_high=np.array([[1, 1]]),
                 heater_resistance=np.array([1]),
                 thermal_link_channels=np.array([[1]]),
                 thermal_link_heatbath=np.array([1]),
                 temperature_heatbath=0,
                 alpha=1,
                 beta=1,
                 gamma=1,
                 s=3,
                 v=60,
                 g=0.001,
                 r=1,
                 n=15,
                 ):

        # input handling
        self.nmbr_channels = len(action_low)
        assert len(action_high) != self.nmbr_channels or \
               len(action_high) != self.nmbr_channels or \
               len(oberservation_high) != self.nmbr_channels, "Actions and observations must have same length!"

        # create action and observation spaces
        self.action_space = spaces.Box(low=action_low.reshape(-1),
                                       high=action_high.reshape(-1),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=oberservation_low.reshape(-1),
                                            high=oberservation_high.reshape(-1),
                                            dtype=np.float32)

        # environment parameters
        self.heater_resistance = heater_resistance
        self.thermal_link_channels = thermal_link_channels
        self.thermal_link_heatbath = thermal_link_heatbath
        self.temperature_heatbath = temperature_heatbath
        self.g = g

        # reward parameters
        self.r = r
        self.s = s
        self.v = v
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n = n

        # sensor parameters
        def check_sensor_pars(k, T0):
            return self.sensor_model(0, k, T0) > 0.1 and \
                   self.sensor_model(0, k, T0) < 0.5 and \
                   self.sensor_model(1, k, T0) > 0.999

        good_pars = False

        while not good_pars:
            k, T0 = np.random.uniform(low=5, high=50), np.random.uniform(low=0, high=1)
            good_pars = check_sensor_pars(k, T0)

        self.k = k
        self.t0 = T0


    def sensor_model(self, T, k, T0):
        return 1/(1 + np.exp(-k(T - T0)))

    def temperature_model(self):
        pass  # TODO

    def reward(self):
        pass  # TODO

    def step(self, action):

        # if we took an action, we were in state 1
        state = 1

        if action == 2:
            reward = 1
        else:
            reward = -1

        # regardless of the action, game is done after a single step
        done = True

        info = {}

        return state, reward, done, info

    def reset(self):
        state = 0  # TODO
        return state

    def render(self, mode='human'):
        pass

    def close(self):
        pass
