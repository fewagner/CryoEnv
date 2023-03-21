import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
import collections
import math


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
                 V_set_iv=(0., 99.),  # first action
                 wait_iv=(2., 100.),  # secound action
                 ph_iv=(0., 0.99),
                 heater_resistance=np.array([100.]),
                 thermal_link_channels=np.array([[1.]]),
                 thermal_link_heatbath=np.array([1.]),
                 temperature_heatbath=0.,
                 min_ph=0.2,
                 g=np.array([0.001]),  # offset from 0
                 T_hyst=np.array([0.1]),
                 T_hyst_reset=np.array([0.9]),
                 hyst_wait=np.array([50]),
                 control_pulse_amplitude=10,
                 env_fluctuations=1,
                 model_pileup_drops=True,
                 prob_drop=np.array([1e-3]),  # per second!
                 prob_pileup=np.array([0.1]),
                 save_trajectory=False,
                 k: np.ndarray = None,  # logistic curve parameters
                 T0: np.ndarray = None,
                 incentive_reset=0,
                 **kwargs,
                 ):

        # input handling
        self.V_set_iv = V_set_iv
        self.ph_iv = ph_iv
        self.wait_iv = wait_iv
        self.k = k
        self.T0 = T0
        self.prob_drop = prob_drop  # TODO for these we need input checks!
        self.prob_pileup = prob_pileup  # TODO for these we need input checks!
        self.model_pileup_drops = model_pileup_drops

        self.nmbr_channels = len(heater_resistance)
        assert k is None or len(k) == self.nmbr_channels, \
            'If k is set, it must have length nmbr_channels!'
        assert T0 is None or len(T0) == self.nmbr_channels, \
            'If T0 is set, it must have length nmbr_channels!'

        assert thermal_link_channels.shape == (
            self.nmbr_channels,
            self.nmbr_channels), \
            "thermal_link_channels must have shape (nmbr_channels, nmbr_channels)!"
        assert len(thermal_link_heatbath) == self.nmbr_channels, \
            "thermal_link_heatbath must have same length as heater_resistance!"

        self.nmbr_actions = 2  # will change later
        self.nmbr_observations = 2  # will change later

        if type(V_set_iv) is tuple:
            assert len(V_set_iv) == 2, \
                "The tuple of V_set_iv must be of length {}.".format(2)
            self.V_set_iv = np.array([V_set_iv for i in range(self.nmbr_channels)])
        elif type(V_set_iv) is list or type(V_set_iv) is np.ndarray:
            assert len(V_set_iv) == self.nmbr_channels, \
                "The list V_set_iv must be of length of {}.".format(self.nmbr_channels)
            self.V_set_iv = V_set_iv
        else:
            raise ValueError("V_set_iv has to be either a tuple or a list/numpy.ndarray.")

        if type(wait_iv) is tuple:
            assert len(wait_iv) == 2, \
                "The tuple of wait_iv must be of length {}.".format(2)
            self.wait_iv = np.array([wait_iv for i in range(self.nmbr_channels)])
        elif type(wait_iv) is list or type(wait_iv) is np.ndarray:
            assert len(wait_iv) == self.nmbr_channels, \
                "The list wait_iv must be of length of {}.".format(self.nmbr_channels)
            self.wait_iv = wait_iv
        else:
            raise ValueError("wait_iv has to be either a tuple or a list/numpy.ndarray.")

        if type(ph_iv) is tuple:
            assert len(ph_iv) == 2, \
                "The tuple of ph_iv must be of length {}.".format(2)
            self.ph_iv = np.array([ph_iv for i in range(self.nmbr_channels)])
        elif type(ph_iv) is list or type(ph_iv) is np.ndarray:
            assert len(ph_iv) == self.nmbr_channels, \
                "The list ph_iv must be of length of {}.".format(self.nmbr_channels)
            self.ph_iv = ph_iv
        else:
            raise ValueError("ph_iv has to be either a tuple or a list/numpy.ndarray.")

        # action_low = np.array([[self.V_set_iv[i][0], self.wait_iv[i][0]] for i in range(self.nmbr_channels)])
        # action_high = np.array([[self.V_set_iv[i][1], self.wait_iv[i][1]] for i in range(self.nmbr_channels)])
        # observation_low = np.array([[self.V_set_iv[i][0], self.ph_iv[i][0]] for i in range(self.nmbr_channels)])
        # observation_high = np.array([[self.V_set_iv[i][1], self.ph_iv[i][1]] for i in range(self.nmbr_channels)])
        #
        # # create action and observation spaces
        # self.action_space = spaces.Box(low=action_low.reshape(-1),
        #                                high=action_high.reshape(-1),
        #                                dtype=np.float32)
        # self.observation_space = spaces.Box(low=observation_low.reshape(-1),
        #                                     high=observation_high.reshape(-1),
        #                                     dtype=np.float32)

        self.action_space = spaces.Box(low=- np.ones(self.nmbr_actions * self.nmbr_channels),
                                       high=np.ones(self.nmbr_actions * self.nmbr_channels),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=- np.ones(self.nmbr_observations * self.nmbr_channels),
                                            high=np.ones(self.nmbr_observations * self.nmbr_channels),
                                            dtype=np.float32)

        # environment parameters
        self.heater_resistance = np.array(heater_resistance)
        self.thermal_link_channels = np.array(thermal_link_channels)
        self.thermal_link_heatbath = np.array(thermal_link_heatbath)
        self.temperature_heatbath = np.array(temperature_heatbath)
        self.g = np.full(self.nmbr_channels, g)
        self.T_hyst = np.full(self.nmbr_channels, T_hyst)
        self.control_pulse_amplitude = np.full(self.nmbr_channels, control_pulse_amplitude)
        self.env_fluctuations = np.full(self.nmbr_channels, env_fluctuations)
        self.T = np.zeros(self.nmbr_channels)
        self.hyst = np.full(self.nmbr_channels, False)  # setting if hysteresis is active
        self.hyst_waited = np.zeros(self.nmbr_channels)
        self.T_hyst_reset = np.full(self.nmbr_channels, T_hyst_reset)
        self.hyst_wait = np.full(self.nmbr_channels, hyst_wait)
        self.incentive_reset = incentive_reset

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
        self.state = self.denorm_state(self.reset())

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
        pile_up = np.zeros(self.nmbr_channels)
        if self.model_pileup_drops:
            piled_up = np.random.uniform(size=self.nmbr_channels) < self.prob_pileup
            pile_up[piled_up] = np.random.exponential(size=np.sum(piled_up),
                                                      scale=self.env_fluctuations * 40)
        T_inj = self.temperature_model(P_R=P_R_inj,
                                       P_E=self.environment_model(V_set) + P_E_long + pile_up)
        height_signal = self.sensor_model(T_inj, self.k, self.T0)

        # difference is pulse height
        phs = np.maximum(height_signal - height_baseline, self.g)

        # hysteresis case
        phs[self.hyst] = self.g[self.hyst]

        return phs, T, T_inj

    def reward(self, new_state, action):

        reward = 0

        # unpack action
        _, wait = action.reshape((self.nmbr_channels, self.nmbr_actions)).T

        # unpack new state
        V_set, ph = new_state.reshape((self.nmbr_channels, self.nmbr_observations)).T

        for w, p, v in zip(wait, ph, V_set):

            # one second for sending a control pulse
            reward -= 1

            # check stability
            stable = p > self.min_ph

            # print(r, dv, w, v, p)

            if stable:
                reward += p * w
            else:
                reward -= w

            reward += self.incentive_reset * v

        return reward

    def norm_action(self, action):
        normed_action = np.copy(action)
        normed_action[0::2] = 2 * (normed_action[0::2] - self.V_set_iv[:, 0]) / self.V_set_iv[:,
                                                                                1] - 1  # norm the v sets
        normed_action[1::2] = 2 * (normed_action[1::2] - self.wait_iv[:, 0]) / self.wait_iv[:, 1] - 1  # norm the waits
        return normed_action

    def denorm_action(self, normed_action):
        action = np.copy(normed_action)
        action[0::2] = self.V_set_iv[:, 1] * (1 + action[0::2]) / 2 + self.V_set_iv[:, 0]  # denorm the v sets
        action[1::2] = self.wait_iv[:, 1] * (1 + action[1::2]) / 2 + self.wait_iv[:, 0]  # denorm the waits
        return action

    def norm_state(self, state):
        normed_state = np.copy(state)
        normed_state[0::2] = 2 * (normed_state[0::2] - self.V_set_iv[:, 0]) / self.V_set_iv[:, 1] - 1  # norm the v sets
        normed_state[1::2] = 2 * (normed_state[1::2] - self.ph_iv[:, 0]) / self.ph_iv[:, 1] - 1  # norm the phs
        return normed_state

    def denorm_state(self, normed_state):
        state = np.copy(normed_state)
        state[0::2] = self.V_set_iv[:, 1] * (1 + state[0::2]) / 2 + self.V_set_iv[:, 0]  # denorm the v sets
        state[1::2] = self.ph_iv[:, 1] * (1 + state[1::2]) / 2 + self.ph_iv[:, 0]  # denorm the phs
        return state

    def step(self, action):
        denormed_action = self.denorm_action(action)
        new_state, reward, done, info = self._step(denormed_action)
        normed_new_state = self.norm_state(new_state)
        return normed_new_state, reward, done, info

    def _step(self, action):
        # unpack action
        future_V_sets, wait = action.reshape((self.nmbr_channels, self.nmbr_actions)).T

        # unpack current state
        V_sets, phs = self.state.reshape((self.nmbr_channels, self.nmbr_observations)).T

        # get the next V sets
        if any(future_V_sets < self.V_set_iv[:, 0]):
            future_V_sets[future_V_sets < self.V_set_iv[:, 0]] = self.V_set_iv[future_V_sets < self.V_set_iv[:, 0]][0]

        # hysteresis handling
        self.T = self.temperature_model(P_R=future_V_sets / self.heater_resistance, P_E=0)
        self.hyst_waited[self.T > self.T_hyst_reset] += wait
        self.hyst[self.hyst_waited > self.hyst_wait] = False
        self.hyst[self.T < self.T_hyst] = True
        self.hyst_waited[self.T < self.T_hyst] = 0
        if self.model_pileup_drops:  # drops due to rare, unexpected vibrations, etc
            drop_probabilities = np.array(
                [np.sum([p * (1 - p) ** i for i in range(int(wait[c]))]) for c, p in enumerate(self.prob_drop)])
            dropped = np.random.uniform(size=self.nmbr_channels) < drop_probabilities
            self.hyst[dropped] = True
            self.hyst_waited[dropped] = True

        # get the next phs
        future_phs, future_T, future_T_inj = self.get_pulse_height(future_V_sets)

        # pack new_state
        new_state = np.array([future_V_sets, future_phs]).T.reshape(-1)

        # get the reward
        reward = self.reward(new_state, action)

        # update state
        self.state = new_state

        # save trajectory
        if self.save_trajectory:
            self.rewards_trajectory.append(reward)
            # self.V_decrease_trajectory.append(V_decs)
            self.wait_trajectory.append(wait)
            # self.reset_trajectory.append(resets)
            self.new_V_set_trajectory.append(future_V_sets)
            self.new_ph_trajectory.append(future_phs)
            self.T_trajectory.append(future_T)
            self.T_inj_trajectory.append(future_T_inj)

        # the task is continuing
        done = False

        info = {}

        return new_state, reward, done, info

    def reset(self, is_training: bool = False):
        if is_training:
            future_V_sets = np.random.randint(self.V_set_iv[:, 0], self.V_set_iv[:, 1])
        else:
            future_V_sets = self.V_set_iv[:, 1]
        future_phs, _, _ = self.get_pulse_height(future_V_sets)
        self.state = np.array([future_V_sets, future_phs]).T.reshape(-1)
        self.hyst = np.full(self.nmbr_channels, False)
        self.hyst_waited = np.zeros(self.nmbr_channels)
        self.reset_trajectories()
        return self.norm_state(self.state)

    def get_trajectory(self):
        return np.array(self.rewards_trajectory), \
               np.array(self.V_decrease_trajectory), \
               np.array(self.wait_trajectory), \
               np.array(self.reset_trajectory), \
               np.array(self.new_V_set_trajectory), \
               np.array(self.new_ph_trajectory), \
               np.array(self.T_trajectory), \
               np.array(self.T_inj_trajectory)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
