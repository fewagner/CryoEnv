import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import collections


def action_to_discrete(reset: list, V_decrease: np.ndarray, wait: np.ndarray,
                       wait_iv: tuple = (2, 100), V_iv: tuple = (0, 99), wait_step=2, V_step=1):
    assert all(w % wait_step == 0 for w in wait), "wait has to be multiple of {}".format(wait_step)
    assert all(w >= wait_iv[0] and w <= wait_iv[1] for w in wait), "wait should be between {} and {}".format(*wait_iv)
    assert all(type(r) == bool for r in reset), "reset has to be bool variable"
    assert all(v % V_step == 0 for v in V_decrease), "V_decrease has to be multiple of {}".format(V_step)
    assert all(v >= V_iv[0] and v <= V_iv[1] for v in V_decrease), "V_decrease should be between {} and {}".format(
        *V_iv)

    nmbr_discrete_V = (V_iv[1] - V_iv[0]) / V_step
    nmbr_discrete_wait = (wait_iv[1] - wait_iv[0]) / wait_step
    n_max = nmbr_discrete_V * nmbr_discrete_wait

    ns = []

    for r, v, w in zip(reset, V_decrease, wait):

        V_in_range = (v - V_iv[0]) / V_step  # in (0, nmbr_discrete_V - 1)
        wait_in_range = (w - wait_iv[0]) / wait_step  # in (0, nmbr_discrete_wait - 1)

        if r:
            ns.append(int(nmbr_discrete_V * nmbr_discrete_wait))
        else:
            ns.append(int(V_in_range * nmbr_discrete_wait + wait_in_range))

    return np.sum([n * n_max ** i for i, n in enumerate(ns)])


def action_from_discrete(n, nmbr_channels, wait_iv=(2, 100), V_iv=(0, 99), wait_step=2, V_step=1):
    nmbr_discrete_V = (V_iv[1] - V_iv[0]) / V_step
    nmbr_discrete_wait = (wait_iv[1] - wait_iv[0]) / wait_step
    n_max = nmbr_discrete_V * nmbr_discrete_wait

    assert n % 1 == 0, "n has to be multiple of 1"
    assert n >= 0 and n <= n_max ** nmbr_channels, "n should be between 0 and {}".format(n_max ** nmbr_channels)

    ns = [int(n % n_max ** (i + 1) / n_max ** i) for i in range(nmbr_channels)]
    reset = []
    V_decrease = []
    wait = []

    for n in ns:
        if n == n_max:
            # first action is reset, second V_decrease, third wait
            reset.append(True)
            V_decrease.append(0)
            wait.append(100)
        else:
            reset.append(False)
            V_decrease.append(np.floor(n / nmbr_discrete_wait) * V_step + V_iv[0])
            wait.append((n % nmbr_discrete_V) * wait_step + wait_iv[0])

    return np.array(reset), np.array(V_decrease), np.array(wait)


def observation_to_discrete(V_set: np.ndarray, ph: np.ndarray, V_iv=(0, 99), ph_iv=(0, 0.99), V_step=1, ph_step=0.01):
    assert all(p % ph_step == 0 for p in ph), "ph has to be multiple of {}".format(ph_step)
    assert all(p >= ph_iv[0] and p <= ph_iv[1] for p in ph), "ph should be between {} and {}".format(*ph_iv)
    assert all(v % V_step == 0 for v in V_set), "V_set has to be multiple of {}".format(V_step)
    assert all(v >= V_iv[0] and v <= V_iv[1] for v in V_set), "V_set should be between {} and {}".format(*V_iv)

    nmbr_discrete_V = (V_iv[1] - V_iv[0]) / V_step
    nmbr_discrete_ph = (ph_iv[1] - ph_iv[0]) / ph_step
    n_max = nmbr_discrete_V * nmbr_discrete_ph - 1

    ns = []

    for v, p in zip(V_set, ph):
        V_in_range = (v - V_iv[0]) / V_step  # in (0, nmbr_discrete_V - 1)
        ph_in_range = (p - ph_iv[0]) / ph_step  # in (0, nmbr_discrete_ph - 1)

        ns.append(int(V_in_range * nmbr_discrete_ph + ph_in_range))

    return np.sum([n * n_max ** i for i, n in enumerate(ns)])


def observation_from_discrete(n, nmbr_channels, V_iv=(0, 99), ph_iv=(0, 0.99), V_step=1, ph_step=0.01):
    nmbr_discrete_V = (V_iv[1] - V_iv[0]) / V_step
    nmbr_discrete_ph = (ph_iv[1] - ph_iv[0]) / ph_step
    n_max = nmbr_discrete_V * nmbr_discrete_ph - 1

    assert n % 1 == 0, "n has to be multiple of 1"
    assert n >= 0 and n <= n_max ** nmbr_channels, "n should be between 0 and {}".format(n_max ** nmbr_channels)

    ns = [int(n % n_max ** (i + 1) / n_max ** i) for i in range(nmbr_channels)]
    V_set = []
    pulse_height = []

    for n in ns:
        V_set.append(np.floor(n / nmbr_discrete_ph) * V_step + V_iv[0])
        pulse_height.append((n % nmbr_discrete_V) * ph_step + ph_iv[0])

    # first observation is V_set, second observation is pulse height
    return np.array(V_set), np.array(pulse_height)


class CryoEnvDiscrete_v0(gym.Env):
    """
    Simplified discrete CryoEnv

    Rewards:
    - If pulse height above 0.X mV —> + waiting time, otherwise - waiting time
    - Always -1 for sending pulse

    Actions:
    - Reset yes/no (V_set back to 99, )
    - Decrease V_set by 0 to 99, in 1 steps
    - Waiting time between 2 and 100 seconds, in 2 seconds steps
    - —> action space size 1 + 100*50 = 5001

    States:
    - V_set 100 steps from 0 to 99
    - PH 100 steps from 0 to 0.99
    - —> observation space size 100*100 = 10000
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 V_set_iv=(0, 99),
                 V_set_step=1,
                 ph_iv=(0, 0.99),
                 ph_step=0.01,
                 wait_iv=(2, 100),
                 wait_step=2,
                 heater_resistance=np.array([100.]),
                 thermal_link_channels=np.array([[1.]]),
                 thermal_link_heatbath=np.array([1.]),
                 temperature_heatbath=0.,
                 min_ph=0.2,
                 g=np.array([0.001]),
                 T_hyst=np.array([0.001]),
                 control_pulse_amplitude=10,
                 env_fluctuations=1,
                 save_trajectory=False,
                 ):

        # input handling
        self.V_set_iv = V_set_iv
        self.V_set_step = V_set_step
        self.ph_iv = ph_iv
        self.ph_step = ph_step
        self.wait_iv = wait_iv
        self.wait_step = wait_step
        nmbr_discrete_V = (V_set_iv[1] - V_set_iv[0]) / V_set_step
        nmbr_discrete_ph = (ph_iv[1] - ph_iv[0]) / ph_step
        nmbr_discrete_wait = (wait_iv[1] - wait_iv[0]) / wait_step

        self.nmbr_channels = len(heater_resistance)

        n_action_max = nmbr_discrete_V * nmbr_discrete_wait ** self.nmbr_channels
        n_observation_max = (nmbr_discrete_V * nmbr_discrete_ph - 1) ** self.nmbr_channels

        assert thermal_link_channels.shape == (
            self.nmbr_channels,
            self.nmbr_channels), "thermal_link_channels must have shape (nmbr_channels, nmbr_channels)!"
        assert len(
            thermal_link_heatbath) == self.nmbr_channels, "thermal_link_heatbath must have same length as heater_resistance!"

        self.action_space = spaces.Discrete(n_action_max)
        self.observation_space = spaces.Discrete(n_observation_max)

        # environment parameters
        self.heater_resistance = np.array(heater_resistance)
        self.thermal_link_channels = np.array(thermal_link_channels)
        self.thermal_link_heatbath = np.array(thermal_link_heatbath)
        self.temperature_heatbath = np.array(temperature_heatbath)
        self.g = g
        self.T_hyst = T_hyst
        self.control_pulse_amplitude = control_pulse_amplitude
        self.env_fluctuations = env_fluctuations

        # reward parameters
        self.min_ph = min_ph

        # sensor parameters
        def check_sensor_pars(k, T0):
            return self.sensor_model(0, k, T0) > 0.1 and \
                   self.sensor_model(0, k, T0) < 0.5 and \
                   self.sensor_model(1, k, T0) > 0.999

        k = np.empty(self.nmbr_channels)
        T0 = np.empty(self.nmbr_channels)

        for i in range(self.nmbr_channels):
            good_pars = False
            while not good_pars:
                k[i], T0[i] = np.random.uniform(low=5, high=50), np.random.uniform(low=0, high=1)
                good_pars = check_sensor_pars(k[i], T0[i])

        self.k = k
        self.T0 = T0

        # initial state
        self.state = self.reset()

        # render
        self.save_trajectory = save_trajectory
        self.actions_trajectory = []
        self.new_states_trajectory = []
        self.rewards_trajectory = []

    def sensor_model(self, T, k, T0):
        return 1 / (1 + np.exp(-k * (T - T0)))

    def temperature_model(self, P_R, P_E):
        T = (self.thermal_link_channels * self.temperature_heatbath + P_R + P_E)
        T = np.linalg.inv(np.diag(self.thermal_link_heatbath) + self.thermal_link_channels - np.diag(
            self.thermal_link_channels @ np.ones(self.nmbr_channels))) @ T
        return T.flatten()

    def environment_model(self, state):
        return np.random.normal(loc=0, scale=self.env_fluctuations, size=1)

    def get_pulse_height(self, V_set):
        # get long scale environment fluctuations - we also get short scale fluctuations further down
        P_E_long = self.environment_model(self.state)

        # height without signal
        P_R = V_set / self.heater_resistance  # P...power; voltage goes through square rooter
        T = self.temperature_model(P_R=P_R,
                                   P_E=self.environment_model(self.state) + P_E_long)
        height_baseline = self.sensor_model(T, self.k, self.T0)

        # height with signal
        P_R_inj = np.sqrt(V_set ** 2 + self.control_pulse_amplitude ** 2) / \
                  self.heater_resistance  # P...power; voltage goes through square rooter
        T_inj = self.temperature_model(P_R=P_R_inj,
                                       P_E=self.environment_model(self.state) + P_E_long)
        height_signal = self.sensor_model(T_inj, self.k, self.T0)

        # difference is pulse height
        phs = np.maximum(height_signal - height_baseline, self.g)

        # hysteresis case
        phs[T < self.T_hyst] = self.g

        return phs

    def reward(self, new_state, action):

        reward = 0

        # unpack action
        reset, V_decrease, wait = action_from_discrete(action)

        # unpack new state
        V_set, ph = observation_from_discrete(new_state)

        for r, dv, w, v, p in zip(reset, V_decrease, wait, V_set, ph):

            # one second for sending a control pulse
            reward = - 1

            # check stability
            stable = p > self.min_ph

            if stable:
                reward += w
            else:
                reward -= w

        return reward

    def step(self, action):
        # unpack action
        resets, V_decs, ws = action_from_discrete(action, nmbr_channels=self.nmbr_channels)

        # unpack current state
        V_sets, phs = observation_from_discrete(self.state)

        # get the next V sets
        future_V_sets = V_sets - V_decs
        future_V_sets[future_V_sets < self.V_set_iv[0]] = self.V_set_iv[0]
        future_V_sets[resets] = self.V_set_iv[1]

        # get the next phs
        future_phs = self.get_pulse_height(future_V_sets)

        # pack new_state
        new_state = observation_to_discrete(V_set=future_V_sets, ph=future_phs)

        # get the reward
        reward = self.reward(new_state, action)

        # update state
        self.state = new_state

        # save trajectory
        if self.save_trajectory:
            self.actions_trajectory.append(action)
            self.new_states_trajectory.append(new_state)
            self.rewards_trajectory.append(reward)

        # the task is continuing
        done = False

        info = {}

        return new_state, reward, done, info

    def reset(self):
        future_V_sets = self.V_set_iv[1]
        future_phs = self.get_pulse_height(future_V_sets)
        self.state = observation_to_discrete(V_set=future_V_sets, ph=future_phs)
        return self.state

    def get_trajectory(self):
        return np.array(self.actions_trajectory), \
               np.array(self.new_states_trajectory), \
               np.array(self.rewards_trajectory)

    def render(self, mode='human'):
        pass

    def close(self):
        pass