import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import collections
import math
from ._discretization import *

class CryoEnvContinuous_v0(gym.Env):
    """
    Simplified continuous CryoEnv

    Rewards:
    - If pulse height above 0.X mV —> + waiting time, otherwise - waiting time
    - Always -1 for sending pulse

    Actions:
    - Decrease V_set by 0 to 99, in 1 steps
    - Waiting time between 2 and 100 seconds, in 2 seconds steps
    - —> action space size 1 + 100*50 = 5001

    Oberservation (State):
    - V_set 100 steps from 0 to 99
    - PH 100 steps from 0 to 0.99
    - —> observation space size 100*100 = 10000

    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 V_set_iv=(0., 99.), # first action
                 wait_iv=(2., 100.), # secound action
                 ph_iv=(0., 0.99),
                 heater_resistance=np.array([100.]),
                 thermal_link_channels=np.array([[1.]]),
                 thermal_link_heatbath=np.array([1.]),
                 temperature_heatbath=0.,
                 min_ph=0.2,
                 g=np.array([0.001]), # offset from 0
                 T_hyst=np.array([0.001]),
                 T_hyst_reset=np.array([0.99]),
                 hyst_wait=np.array([50]),
                 control_pulse_amplitude=10,
                 env_fluctuations=1,
                 save_trajectory=False,
                 k: np.ndarray = None, # logistic curve parameters
                 T0: np.ndarray = None,
                 **kwargs,
                 ):

        # input handling
        self.V_set_iv = V_set_iv
        self.ph_iv = ph_iv
        self.wait_iv = wait_iv
        self.k = k
        self.T0 = T0


        self.nmbr_channels = len(heater_resistance)
        assert k is None or len(k) == self.nmbr_channels,
            'If k is set, it must have length nmbr_channels!'
        assert T0 is None or len(T0) == self.nmbr_channels,
            'If T0 is set, it must have length nmbr_channels!'

        assert thermal_link_channels.shape == (
            self.nmbr_channels,
            self.nmbr_channels),
            "thermal_link_channels must have shape (nmbr_channels, nmbr_channels)!"
        assert len(thermal_link_heatbath) == self.nmbr_channels,
            "thermal_link_heatbath must have same length as heater_resistance!"

        self.nmbr_actions = 2 # will change later
        self.nmbr_observations = 2 # will change later

        if type(V_set_iv)==tuple:
            assert len(V_set_iv)==nmbr_actions,
                "The tuple of V_set_iv must be of length {}.".format(nmbr_actions)
            self.V_set_iv = [[V_set_iv] for i in range(nmbr_channels)]
        elif type(V_set_iv)==list or type(V_set_iv)==np.ndarray:
            assert len(V_set_iv)==nmbr_channels,
                "The list V_set_iv must be of length of {}.".format(nmbr_channels)
            self.V_set_iv = V_set_iv
        else:
            raise ValueError("V_set_iv has to be either a tuple or a list/numpy.ndarray.")


        if type(wait_iv)==tuple:
            assert len(wait_iv)==nmbr_actions,
                "The tuple of wait_iv must be of length {}.".format(nmbr_actions)
            self.wait_iv = [[wait_iv] for i in range(nmbr_channels)]
        elif type(wait_iv)==list or type(wait_iv)==np.ndarray):
            assert len(wait_iv)==nmbr_channels,
                "The list wait_iv must be of length of {}.".format(nmbr_channels)
            self.wait_iv = wait_iv
        else:
            raise ValueError("wait_iv has to be either a tuple or a list/numpy.ndarray.")


        if type(ph_iv)==tuple:
            assert len(ph_iv)==nmbr_actions,
                "The tuple of ph_iv must be of length {}.".format(nmbr_actions)
            self.ph_iv = [[ph_iv] for i in range(nmbr_channels)]
        elif type(ph_iv)==list or type(ph_iv)==np.ndarray):
            assert len(ph_iv)==nmbr_channels,
                "The list ph_iv must be of length of {}.".format(nmbr_channels)
            self.ph_iv = ph_iv
        else:
            raise ValueError("ph_iv has to be either a tuple or a list/numpy.ndarray.")


        action_low  = np.array([[self.V_set_iv[i][0], self.wait_iv[i][0]] for i in range(nmbr_channels)])
        action_high = np.array([[self.V_set_iv[i][1], self.wait_iv[i][1]] for i in range(nmbr_channels)])
        observation_low  = np.array([[self.V_set_iv[i][0], self.ph_iv[i][0]] for i in range(nmbr_channels)])
        observation_high = np.array([[self.V_set_iv[i][0], self.ph_iv[i][0]] for i in range(nmbr_channels)])


        # create action and observation spaces
        self.action_space = spaces.Box(low=action_low.reshape(-1),
                                       high=action_high.reshape(-1),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=observation_low.reshape(-1),
                                            high=observation_high.reshape(-1),
                                            dtype=np.float32)


        # environment parameters
        self.heater_resistance = np.array(heater_resistance)
        self.thermal_link_channels = np.array(thermal_link_channels)
        self.thermal_link_heatbath = np.array(thermal_link_heatbath)
        self.temperature_heatbath = np.array(temperature_heatbath)
        self.g = np.full(nmbr_channels, g)
        self.T_hyst = np.full(nmbr_channels, T_hyst)
        self.control_pulse_amplitude = np.full(nmbr_channels, control_pulse_amplitude)
        self.env_fluctuations = np.full(nmbr_channels, env_fluctuations)

        self.hyst = np.full(nmbr_channels, False) # setting if hysteresis is active
        self.hyst_waited = np.zeros(nmbr_channels)
        self.T_hyst_reset = np.full(nmbr_channels, T_hyst_reset)
        self.hyst_wait = np.full(nmbr_channels, hyst_wait)


        # reward parameters
        self.min_ph = min_ph

        # sensor parameters
        def check_sensor_pars(k, T0):
            return self.sensor_model(0, k, T0) > 0.0001 and \
                   self.sensor_model(0, k, T0) < 0.01 and \
                   self.sensor_model(1, k, T0) > 0.95

        k = np.empty(self.nmbr_channels)
        T0 = np.empty(self.nmbr_channels)

        for i in range(self.nmbr_channels):
            good_pars = False
            while not good_pars:
                k[i] = np.random.uniform(low=3, high=15)
                T0[i] = np.random.uniform(low=0, high=1)
                good_pars = check_sensor_pars(k[i], T0[i])

        if self.k is None:
            self.k = k
        if self.T0 is None:
            self.T0 = T0

        # initial state
        self.state = self.reset()

        # render
        self.save_trajectory = save_trajectory
        self.reset_trajectories()

    def reset_trajectories(self):
        self.rewards_trajectory = []
        self.V_decrease_trajectory = []
        self.wait_trajectory = []
        self.reset_trajectory = []
        self.new_V_set_trajectory = []
        self.new_ph_trajectory = []
        self.T_trajectory = []
        self.T_inj_trajectory = []


    def sensor_model(self, T, k, T0):
        return 1 / (1 + np.exp(-k * (T - T0)))

    def temperature_model(self, P_R, P_E):
        T = (self.thermal_link_channels * self.temperature_heatbath + P_R + P_E)
        T = np.linalg.inv(np.diag(self.thermal_link_heatbath) + self.thermal_link_channels - np.diag(
            self.thermal_link_channels @ np.ones(self.nmbr_channels))) @ T
        return T.flatten()

    def environment_model(self, V_set):
        return np.random.normal(loc=0, scale=self.env_fluctuations, size=self.nmbr_channels)

    def get_pulse_height(self, V_set):
        # get long scale environment fluctuations - we also get short scale fluctuations further down
        P_E_long = self.environment_model(V_set)

        # height without signal
        P_R = V_set / self.heater_resistance  # P...power; voltage goes through square rooter
        T = self.temperature_model(P_R=P_R,
                                   P_E=self.environment_model(V_set) + P_E_long)
        height_baseline = self.sensor_model(T, self.k, self.T0)

        # height with signal
        P_R_inj = np.sqrt(V_set ** 2 + self.control_pulse_amplitude ** 2) / \
                  self.heater_resistance  # P...power; voltage goes through square rooter
        T_inj = self.temperature_model(P_R=P_R_inj,
                                       P_E=self.environment_model(V_set) + P_E_long)
        height_signal = self.sensor_model(T_inj, self.k, self.T0)

        # difference is pulse height
        phs = np.maximum(height_signal - height_baseline, self.g)

        # hysteresis case

        phs[self.hyst] = self.g[self.hyst]

        return phs, T, T_inj

    def reward(self, new_state, action):

        reward = 0

        # unpack action
        V_decrease, wait = action.reshape((self.nmbr_channels, self.nmbr_actions)).T

        # unpack new state
        V_set, ph = new_state.reshape((self.nmbr_channels, self.nmbr_observations)).T

        for r, dv, w, p in zip(reset, V_decrease, V_set, ph):

            # one second for sending a control pulse
            reward = - 1

            # check stability
            stable = p > self.min_ph

            # print(r, dv, w, v, p)

            if stable:
                reward += wait
            else:
                reward -= wait

        return reward

    def step(self, action):
        # unpack action
        future_V_sets, wait = action.reshape((self.nmbr_channels, self.nmbr_actions)).T

        # unpack current state
        V_sets, phs = self.state

        # get the next V sets
        future_V_sets[future_V_sets < self.V_set_iv[0]] = self.V_set_iv[future_V_sets < self.V_set_iv[0]][0]
        future_V_sets[resets] = self.V_set_iv[resets][1]

        self.hyst_waited += wait
        self.hyst = np.logical_and(
                        np.logical_or(T < self.T_hyst, self.hyst),
                        self.hyst_waited < self.hyst_wait
                    )
        self.hyst_waited[self.hyst] = np.zeros(self.hyst_waited[self.hyst].shape)


        # get the next phs
        future_phs, future_T, future_T_inj = self.get_pulse_height(future_V_sets)

        # pack new_state
        new_state = np.array([future_V_sets, future_phs]).T

        # get the reward
        reward = self.reward(new_state, action)

        # update state
        self.state = new_state

        # save trajectory
        if self.save_trajectory:
            self.rewards_trajectory.append(reward)
            self.V_decrease_trajectory.append(V_decs)
            self.wait_trajectory.append(wait)
            self.reset_trajectory.append(resets)
            self.new_V_set_trajectory.append(future_V_sets)
            self.new_ph_trajectory.append(future_phs)
            self.T_trajectory.append(future_T)
            self.T_inj_trajectory.append(future_T_inj)

        # the task is continuing
        done = False

        info = {}

        return new_state, reward, done, info

    def reset(self):
        future_V_sets = self.V_set_iv[1]*np.ones([self.nmbr_channels], dtype=float)
        future_phs, _, _ = self.get_pulse_height(future_V_sets)
        self.state = self.observation_to_discrete(V_set=future_V_sets, ph=future_phs)
        self.hyst = np.full(nmbr_channels, False)
        self.hyst_waited = np.zeros(nmbr_channels)
        self.reset_trajectories()
        return self.state

    def get_trajectory(self):
        return np.array(self.rewards_trajectory), \
               np.array(self.V_decrease_trajectory), \
               np.array(self.wait_trajectory), \
               np.array(self.reset_trajectory),\
               np.array(self.new_V_set_trajectory), \
               np.array(self.new_ph_trajectory), \
               np.array(self.T_trajectory), \
               np.array(self.T_inj_trajectory)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
