import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import collections


def action_to_discrete(reset, V_decrease, wait, wait_iv=(2, 100), V_iv=(0, 99), wait_step=2, V_step=1):
    assert wait % wait_step == 0, "wait has to be multiple of {}".format(wait_step)
    assert wait >= wait_iv[0] and wait <= wait_iv[1], "wait should be between {} and {}".format(*wait_iv)
    assert type(reset) == bool, "reset has to be bool variable"
    assert V_decrease % V_step == 0, "V_decrease has to be multiple of {}".format(V_step)
    assert V_decrease >= V_iv[0] and V_decrease <= V_iv[1], "V_decrease should be between {} and {}".format(*V_iv)

    nmbr_discrete_V = (V_iv[1] - V_iv[0]) / V_step
    nmbr_discrete_wait = (wait_iv[1] - wait_iv[0]) / wait_step
    V_in_range = (V_decrease - V_iv[0]) / V_step  # in (0, nmbr_discrete_V - 1)
    wait_in_range = (wait - wait_iv[0]) / wait_step  # in (0, nmbr_discrete_wait - 1)

    if reset:
        return int(nmbr_discrete_V * nmbr_discrete_wait)
    else:
        return int(V_in_range * nmbr_discrete_wait + wait_in_range)


def action_from_discrete(n, wait_iv=(2, 100), V_iv=(0, 99), wait_step=2, V_step=1):
    nmbr_discrete_V = (V_iv[1] - V_iv[0]) / V_step
    nmbr_discrete_wait = (wait_iv[1] - wait_iv[0]) / wait_step
    n_max = nmbr_discrete_V * nmbr_discrete_wait

    assert n % 1 == 0, "n has to be multiple of 1"
    assert n >= 0 and n <= n_max, "n should be between 0 and {}".format(n_max)

    if n == n_max:
        return (True, 0, 100)  # first action is reset, second V_decrease, third wait
    else:
        return (False, np.floor(n / nmbr_discrete_wait) * V_step + V_iv[0] , (n % nmbr_discrete_V) * wait_step + wait_iv[0])


def observation_to_discrete(V_set, ph, V_iv=(0, 99), ph_iv=(0, 0.99), V_step=1, ph_step=0.01):
    assert ph % ph_step == 0, "ph has to be multiple of {}".format(ph_step)
    assert ph >= ph_iv[0] and ph <= ph_iv[1], "ph should be between {} and {}".format(*ph_iv)
    assert V_set % V_step == 0, "V_set has to be multiple of {}".format(V_step)
    assert V_set >= V_iv[0] and V_set <= V_iv[1], "V_set should be between {} and {}".format(*V_iv)

    nmbr_discrete_V = (V_iv[1] - V_iv[0]) / V_step
    nmbr_discrete_ph = (ph_iv[1] - ph_iv[0]) / ph_step
    V_in_range = (V_set - V_iv[0]) / V_step  # in (0, nmbr_discrete_V - 1)
    ph_in_range = (ph - ph_iv[0]) / ph_step  # in (0, nmbr_discrete_ph - 1)

    return int(V_in_range * nmbr_discrete_ph + ph_in_range)

def observation_from_discrete(n, V_iv=(0, 99), ph_iv=(0, 0.99), V_step=1, ph_step=0.01):
    nmbr_discrete_V = (V_iv[1] - V_iv[0]) / V_step
    nmbr_discrete_ph = (ph_iv[1] - ph_iv[0]) / ph_step
    n_max = nmbr_discrete_V * nmbr_discrete_ph

    assert n % 1 == 0, "n has to be multiple of 1"
    assert n >= 0 and n <= n_max, "n should be between 0 and {}".format(n_max)

    # first observation is V_set, second observation is pulse height
    return (np.floor(n / nmbr_discrete_ph) * V_step + V_iv[0], (n % nmbr_discrete_V) * ph_step + ph_iv[0])

class CryoEnvDiscrete_v0(gym.Env):
    """
    Simplified discrete CryoEnv

    Rewards:
    - If pulse height above 0.5 mV —> + waiting time, otherwise - waiting time
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
                 heater_resistance=np.array([100.]),
                 thermal_link_channels=np.array([[1.]]),
                 thermal_link_heatbath=np.array([1.]),
                 temperature_heatbath=0.,
                 # alpha=1.,
                 # beta=1.,
                 # gamma=1.,
                 s=3.,
                 v=60.,
                 g=0.001,
                 r=1.,
                 n=15,
                 control_pulse_amplitude=10,
                 env_fluctuations=1,
                 save_trajectory=False,
                 ):

        # input handling
        self.max_vset = action_high[0, 1]
        self.nmbr_channels = len(action_low)
        self.nmbr_actions = 3  # first action is V_set decrease, second is waiting time, third is reset prob
        self.nmbr_observations = 2  # first observation is V_set, second is PH
        assert action_high.shape == (
            self.nmbr_channels, self.nmbr_actions), "action_high must have same shape as action_low!"
        assert oberservation_low.shape == (
            self.nmbr_channels, self.nmbr_observations), "oberservation_low must have same length as action_low!"
        assert oberservation_high.shape == (
            self.nmbr_channels, self.nmbr_observations), "oberservation_high must have same shape as oberservation_low!"
        assert len(heater_resistance) == self.nmbr_channels, "heater_resistance must have same length as action_low!"
        assert thermal_link_channels.shape == (
            self.nmbr_channels,
            self.nmbr_channels), "thermal_link_channels must have shape (nmbr_channels, nmbr_channels)!"
        assert len(
            thermal_link_heatbath) == self.nmbr_channels, "thermal_link_heatbath must have same length as action_low!"

        # create action and observation spaces
        # self.action_space = spaces.Box(low=action_low.reshape(-1),
        #                                high=action_high.reshape(-1),
        #                                dtype=np.float32)
        # self.observation_space = spaces.Box(low=oberservation_low.reshape(-1),
        #                                     high=oberservation_high.reshape(-1),
        #                                     dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(5001)
        self.observation_space = spaces.Discrete(low=oberservation_low.reshape(-1),
                                                 high=oberservation_high.reshape(-1),
                                                 dtype=np.float32)

        # environment parameters
        self.heater_resistance = np.array(heater_resistance)
        self.thermal_link_channels = np.array(thermal_link_channels)
        self.thermal_link_heatbath = np.array(thermal_link_heatbath)
        self.temperature_heatbath = np.array(temperature_heatbath)
        self.g = g
        self.hysteresis = np.zeros(self.nmbr_channels, dtype=bool)
        self.control_pulse_amplitude = control_pulse_amplitude
        self.env_fluctuations = env_fluctuations

        # reward parameters
        self.r = r
        self.s = s
        self.v = v
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.last_phs = collections.deque(maxlen=self.n)

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

    def reward(self, new_state, action):

        reward = 0

        for (st, a) in zip(new_state.reshape(-1, self.nmbr_observations),
                           action.reshape(-1, self.nmbr_actions)):

            # unpack action
            dV = a[0]
            w = a[1]
            z = a[2]

            # unpack new state
            V_set_new = st[0]
            ph_new = st[1]

            # check stability
            if len(self.last_phs) >= self.n:
                if np.abs(ph_new - np.mean(self.last_phs)) > self.s * np.std(self.last_phs) \
                        or ph_new < 2 * self.g:
                    stable = False
                else:
                    stable = True
                    self.last_phs.append(ph_new)
            else:
                stable = True
                self.last_phs.append(ph_new)

            # elapsed time
            time = np.copy(w)

            # reset case
            if z > 0.5:
                reward -= self.gamma  # detector needs to go back to normal conducting phase
                time = self.v

            # normal case
            # else:
            reward -= self.alpha / time / ph_new  # detector range maximization
            if len(self.last_phs) > self.n:
                reward -= self.beta / time * np.std(self.last_phs)  # detector sigma minimization
            reward -= self.gamma * (1 / time + self.r * dV)  # dead time due to sending of pulse and ramping
            if not stable:
                reward -= self.gamma  # penalty for instability

        return reward

    def step(self, action):

        # get the next state
        new_state = np.empty((self.nmbr_channels, 2), dtype=self.state.dtype)

        for c, (st, a) in enumerate(
                zip(self.state.reshape(-1, self.nmbr_observations), action.reshape(-1, self.nmbr_actions))):

            # unpack action
            dV = a[0]
            w = a[1]
            z = a[2]

            # unpack state
            V_set = st[0]
            ph = st[1]

            # reset case
            if z > 0.5:
                new_state[c, :] = np.array([self.max_vset, self.g])
                self.hysteresis[c] = False
            else:

                # new Vset
                new_state[c, 0] = V_set - dV

                # new ph
                if self.hysteresis[c]:
                    new_state[c, 1] = self.g
                else:

                    # get long scale environment fluctuations
                    P_E_long = self.environment_model(self.state)

                    # height without signal
                    P_R = new_state[c, 0] / self.heater_resistance[c]  # voltage goes through square rooter
                    T = self.temperature_model(P_R=P_R,
                                               P_E=self.environment_model(self.state) + P_E_long)
                    height_baseline = self.sensor_model(T, self.k[c], self.T0[c])

                    # height with signal
                    P_R_inj = np.sqrt(new_state[c, 0] ** 2 + self.control_pulse_amplitude ** 2) / \
                              self.heater_resistance[c]  # voltage goes through square rooter
                    T_inj = self.temperature_model(P_R=P_R_inj,
                                                   P_E=self.environment_model(self.state) + P_E_long)
                    height_signal = self.sensor_model(T_inj, self.k[c], self.T0[c])

                    # difference is pulse height
                    new_state[c, 1] = np.maximum(height_signal - height_baseline, self.g)

        # get the reward
        reward = self.reward(new_state, action)

        # update state
        new_state = new_state.reshape(-1)
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
        self.state = np.array([[self.max_vset, self.g] * self.nmbr_channels]).reshape(-1)
        self.hysteresis[:] = False
        return self.state

    def get_trajectory(self):
        return np.array(self.actions_trajectory), \
               np.array(self.new_states_trajectory), \
               np.array(self.rewards_trajectory)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
