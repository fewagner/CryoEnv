import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
import collections
import math
import matplotlib.pyplot as plt
from ..cryosig._detector_model import DetectorModel
from ..cryosig._parameter_sampler import sample_parameters
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import display


class CryoEnvSigWrapper(gym.Env):
    """
    TODO
    """

    metadata = {'render_modes': ['human', 'mpl', 'plotly']}

    def __init__(self, pars=None, omega=1e-2, render_mode=None,  # sample_pars=False,
                 rand_start=False, relax_time=90, log_reward=False, tpa_in_state=True,
                 div_adc_by_bias=True, rand_tpa=True,
                 ):
        if pars is not None:
            self.pars = pars
        else:
            self.pars = {}
        # if sample_pars:
        #     self.pars = sample_parameters(**self.pars)
        self.detector = DetectorModel(**self.pars, verb=False)
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
        self.div_adc_by_bias = div_adc_by_bias
        # self.dac_memory = np.zeros(self.detector.nmbr_heater)
        # self.Ib_memory = np.zeros(self.detector.nmbr_tes)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.rand_tpa = rand_tpa

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
                              tpa=self.detector.tpa_queue[self.detector.tpa_idx])
        if not self.rand_tpa:
            self.detector.tpa_idx += 1
            if self.detector.tpa_idx + 1 > len(self.detector.tpa_queue):
                self.detector.tpa_idx = 0
        else:
            self.detector.tpa_idx = np.random.choice(len(self.detector.tpa_queue))

        new_state = np.ones(self.nmbr_observations)
        new_state[:self.ntes] = self.detector.get('ph', norm=True, div_adc_by_bias=self.div_adc_by_bias)
        new_state[self.ntes:2 * self.ntes] = self.detector.get('rms', norm=True, div_adc_by_bias=self.div_adc_by_bias)
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

        if self.render_mode == "plotly":  # render before CP is sent
            self.render()

        # attention, important that we take the reward here, otherwise we would calc it with the CPs
        if not self.log_reward:
            # exclude the TPA from the reward for importance sampling of small TPAs!
            reward = - np.sum(
                self.detector.rms / self.detector.ph)  # * self.detector.tpa, np.maximum(self.detector.ph, self.detector.rms)
        else:
            reward = - np.log(
                np.sum(self.detector.rms * self.detector.tpa / np.maximum(self.detector.ph, self.detector.rms)))
        reward -= self.omega * np.sum((new_state[2 * self.ntes:3 * self.ntes + self.nheater] -
                                       self.state[2 * self.ntes:3 * self.ntes + self.nheater]) ** 2)

        # add the CPHs to the state
        self.detector.wait(seconds=self.detector.tp_interval - self.detector.t[-1])
        self.detector.trigger(er=np.zeros(self.detector.nmbr_components),
                              tpa=10. * np.ones(self.nheater))
        new_state[-self.nheater:] = self.detector.get('ph', norm=True, div_adc_by_bias=self.div_adc_by_bias)

        self.state = new_state

        terminated = False
        truncated = False

        return new_state, reward, terminated, truncated, info

    def reset(self, clear_buffer=False):

        info = {}

        if not self.rand_start:
            if clear_buffer:
                self.detector.clear_buffer()
            self.detector.set_control(dac=-np.ones(self.detector.nmbr_heater),
                                      Ib=-np.ones(self.detector.nmbr_tes),
                                      norm=True)
            self.state = - np.ones(self.nmbr_observations)

        else:
            info = {}

            if clear_buffer:
                self.detector.clear_buffer()
            action = self.action_space.sample()
            randflag = np.random.choice([True, False], size=self.nmbr_actions)
            action[randflag] = np.random.choice([-1, 1], size=np.sum(randflag))  # put to edge of parameter space
            self.detector.tpa_idx = np.random.choice(len(self.detector.tpa_queue))

            self.detector.set_control(dac=action[:self.nheater],
                                      Ib=action[self.nheater:self.nheater + self.ntes],
                                      norm=True)
            self.detector.wait(seconds=2 * self.relax_time)
            self.detector.trigger(er=np.zeros(self.detector.nmbr_components),
                                  tpa=self.detector.tpa_queue[self.detector.tpa_idx])
            if not self.rand_tpa:
                self.detector.tpa_idx += 1
                if self.detector.tpa_idx + 1 > len(self.detector.tpa_queue):
                    self.detector.tpa_idx = 0
            else:
                self.detector.tpa_idx = np.random.choice(len(self.detector.tpa_queue))

            new_state = np.ones(self.nmbr_observations)
            new_state[:self.ntes] = self.detector.get('ph', norm=True, div_adc_by_bias=self.div_adc_by_bias)
            new_state[self.ntes:2 * self.ntes] = self.detector.get('rms', norm=True,
                                                                   div_adc_by_bias=self.div_adc_by_bias)
            new_state[2 * self.ntes:3 * self.ntes] = self.detector.get('Ib', norm=True)
            new_state[3 * self.ntes:3 * self.ntes + self.nheater] = self.detector.get('dac', norm=True)
            if self.tpa_in_state:
                new_state[3 * self.ntes + self.nheater:3 * self.ntes + 2 * self.nheater] = self.detector.get('tpa',
                                                                                                             norm=True)
                # new_state[3 * self.ntes + 2 * self.nheater:4 * self.ntes + 2 * self.nheater] = self.detector.get('Ib',
                #                                                                                                  norm=True)
                # new_state[4 * self.ntes + 2 * self.nheater:4 * self.ntes + 3 * self.nheater] = self.detector.get('dac',
                #                                                                                                  norm=True)

            # add the CPHs to the state
            self.detector.wait(seconds=self.detector.tp_interval - self.detector.t[-1])
            self.detector.trigger(er=np.zeros(self.detector.nmbr_components),
                                  tpa=10. * np.ones(self.nheater))
            new_state[-self.nheater:] = self.detector.get('ph', norm=True, div_adc_by_bias=self.div_adc_by_bias)

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

        elif self.render_mode == "plotly":
            if self.display is not None:
                for i in range(self.ntes):
                    pulse = self.detector.buffer_pulses[-self.ntes:]
                    pulse = pulse[i]
                    flag = np.array(self.detector.buffer_channel) == i
                    buffer_ph = np.array(self.detector.buffer_ph)[flag]
                    idx_start = 0 if len(flag) % 2 == 0 else 1  # this assumes every second pulse is CP
                    ph = buffer_ph[idx_start::2]
                    buffer_dac = np.array(self.detector.buffer_dac)[flag]
                    dac = buffer_dac[idx_start::2]
                    buffer_Ib = np.array(self.detector.buffer_Ib)[flag]
                    ib = buffer_Ib[idx_start::2]
                    self.update_display(pulse, ph, dac, ib, tes_channel=i)

    def launch_display(self, title=None, color=None):

        subplot_titles = []
        for i in range(self.ntes):
            subplot_titles.extend(
                ["Pulse TES {}".format(i), "PH TES {}".format(i), "DAC TES {}".format(i), "IB TES {}".format(i)])

        fig = make_subplots(rows=2 * self.ntes, cols=2, subplot_titles=subplot_titles)

        for i in range(self.ntes):
            fig.add_trace(go.Scatter(y=[0.], mode="lines", name="Pulse TES {}".format(i), marker=dict(color=color)),
                          row=2 * i + 1, col=1)
            fig.add_trace(go.Scatter(y=[0.], mode="lines", name="PH TES {}".format(i), marker=dict(color=color)),
                          row=2 * i + 1,
                          col=2)
            fig.add_trace(go.Scatter(y=[0.], mode="lines", name="DAC TES {}".format(i), marker=dict(color=color)),
                          row=2 * i + 2,
                          col=1)
            fig.add_trace(go.Scatter(y=[0.], mode="lines", name="IB TES {}".format(i), marker=dict(color=color)),
                          row=2 * i + 2,
                          col=2)

        self.display = go.FigureWidget(fig)
        self.display.update_layout(template="plotly_dark")
        self.display.update_layout(height=self.ntes * 400)
        self.display.layout.title = title
        self.display.layout.showlegend = False
        display(self.display)

    def update_display(self, pulse, ph, dac, ib, tes_channel=0):
        if self.display is not None:
            self.display.data[4 * tes_channel + 0]['y'] = pulse
            self.display.data[4 * tes_channel + 1]['y'] = ph
            self.display.data[4 * tes_channel + 2]['y'] = dac
            self.display.data[4 * tes_channel + 3]['y'] = ib

    def detach_display(self):
        if self.display is not None:
            self.display = None

    def close(self):
        _ = self.reset(clear_buffer=True)
