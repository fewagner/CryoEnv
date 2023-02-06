import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.constants import e, k
from scipy.signal import butter, freqs
import numba as nb
import matplotlib.pyplot as plt
import time
from scipy.optimize import fsolve
from tqdm.auto import tqdm, trange
import pdb
from numbalsoda import lsoda_sig, lsoda
import pandas as pd
from ._transition_curves import Rt_smooth, Rt_kinky

class DetectorModule:
    """
    TODO
    """

    def __init__(self,
                 record_length=16384,
                 sample_frequency=25000,
                 C=None,  # pJ / mK, is defined later bec mutable
                 Gb=np.array([5e-3, 5e-3]),  # pW / mK
                 G=np.array([[0., 1e-3], [1e-3, 0.], ]),  # heat cond between components, pW / mK
                 lamb=0.003,  # thermalization time (s)
                 eps=np.array([[0.99, 0.01], [0.1, 0.9], ]),
                 # share thermalization in components
                 delta=np.array([[0.02, 0.98], ]),
                 # share thermalization in components
                 Rs=np.array([0.035]),  # Ohm
                 Rh=np.array([10]),  # Ohm
                 L=np.array([3.5e-7]),  # H
                 Rt0=np.array([0.2]),  # Ohm
                 k=np.array([2.]),  # 1/mK
                 Tc=np.array([15.]),  # mK
                 Ib=np.array([1.]),  # muA
                 dac=np.array([0.]),  # V
                 pulser_scale=np.array([1.]),  # scale factor
                 heater_attenuator=np.array([1.]),
                 tes_flag=np.array([True, False], dtype=bool),  # which component is a tes
                 heater_flag=np.array([False, True], dtype=bool),  # which component has a heater
                 t0=.16,  # onset of the trigger, s
                 pileup_prob=0.05,  # percent / record window
                 pileup_comp=1,
                 tpa_queue=np.array([0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ]),  # V
                 tp_interval=5,  # s
                 max_buffer_len=10000,
                 dac_ramping_speed=np.array([2e-3]),  # V / s
                 Ib_ramping_speed=np.array([5e-3]),  # muA / s
                 xi=np.array([1.]),  # squid conversion current to voltage, V / muA
                 i_sq=np.array([2 * 1e-12]),  # squid noise, A / sqrt(Hz)
                 tes_fluct=np.array([2e-4]),  # ??
                 emi=np.array([2e-10]),  # ??
                 lowpass=1e4,  # Hz
                 Tb=None,  # function that takes one positional argument t, returns Tb
                 Rt=None,
                 # function that takes one positional argument T, returns Rt
                 which_curve='Rt_smooth',
                 tau=np.array([10]),
                 dac_range=(0., 5.),  # V
                 Ib_range=(0, 1e1),  # muA
                 adc_range=(-10., 10.),  # V
                 store_raw=True,
                 ):

        if C is None:
            self.C = np.array([5e-5, 5e-4])
        else:
            self.C = C

        if Rt is not None:
            self.Rt = Rt
        else:
            self.Rt = [self.Rt_init(which_curve, k_, Tc_, Rt0_) for k_, Tc_, Rt0_ in zip(k, Tc, Rt0)]

        tpa_queue = np.array(tpa_queue)

        # define number of thermal components
        self.nmbr_components = len(self.C)
        assert len(tes_flag) == self.nmbr_components, ''
        assert len(heater_flag) == self.nmbr_components, ''
        assert len(self.C) == self.nmbr_components, ''
        assert len(G) == self.nmbr_components, ''
        assert len(G[0]) == self.nmbr_components, ''
        assert len(Gb) == self.nmbr_components, ''
        assert eps.shape[0] == eps.shape[1] == self.nmbr_components, ''
        assert delta.shape[1] == self.nmbr_components, ''
        assert pileup_comp < self.nmbr_components, ''

        # define number of tes
        self.nmbr_tes = len(Rs)
        assert len(Rs) == self.nmbr_tes, ''
        assert np.sum(tes_flag) == self.nmbr_tes, ''
        assert len(L) == self.nmbr_tes, ''
        assert len(Ib_ramping_speed) == self.nmbr_tes, ''
        assert len(i_sq) == self.nmbr_tes, ''
        assert len(tes_fluct) == self.nmbr_tes, ''
        assert len(emi) == self.nmbr_tes, ''
        assert len(Rt0) == self.nmbr_tes, ''
        assert len(k) == self.nmbr_tes, ''
        assert len(Tc) == self.nmbr_tes, ''
        assert len(Ib) == self.nmbr_tes, ''
        assert len(self.Rt) == self.nmbr_tes, ''
        if Rt is None:
            assert len(Rt0) == self.nmbr_tes, ''
            assert len(k) == self.nmbr_tes, ''
            assert len(Tc) == self.nmbr_tes, ''
            assert len(Ib) == self.nmbr_tes, ''
        else:
            assert len(Rt) == self.nmbr_tes, ''
            assert Rt[0][np.array([0.01, 0.02])].shape == (2, ), ''

        # define number of heaters
        self.nmbr_heater = len(Rh)
        assert len(Rh) == self.nmbr_heater, ''
        assert np.sum(heater_flag) == self.nmbr_heater, ''
        assert len(pulser_scale) == self.nmbr_heater, ''
        assert len(heater_attenuator) == self.nmbr_heater, ''
        assert len(dac) == self.nmbr_heater, ''
        assert len(dac_ramping_speed) == self.nmbr_heater, ''
        assert len(tau) == self.nmbr_heater, ''
        assert len(tpa_queue.shape) == 1 or (len(tpa_queue.shape) == 2 and tpa_queue.shape[1] == self.nmbr_heater), ''

        if len(tpa_queue.shape) == 1:
            tpa_queue = np.repeat(tpa_queue.reshape(-1, 1), self.nmbr_heater, axis=1)

        self.Gb = Gb
        self.G = G
        self.lamb = lamb
        self.eps = eps
        self.delta = delta
        self.Rs = Rs
        self.Rh = Rh
        self.Rt0 = Rt0
        self.L = L
        self.k = k
        self.Tc = Tc
        self.Ib = Ib
        self.dac = dac
        self.U_sq_Rh = np.copy(dac)
        self.tau = tau
        self.pulser_scale = pulser_scale
        self.heater_attenuator = heater_attenuator
        self.tes_flag = tes_flag
        self.heater_flag = heater_flag
        self.record_length = record_length
        self.sample_frequency = sample_frequency
        self.t0 = t0
        self.pileup_prob = pileup_prob
        self.pileup_comp = pileup_comp
        self.tpa_queue = tpa_queue
        self.tp_interval = tp_interval
        self.max_buffer_len = max_buffer_len
        self.dac_ramping_speed = dac_ramping_speed
        self.Ib_ramping_speed = Ib_ramping_speed
        self.xi = xi
        self.i_sq = i_sq
        self.tes_fluct = tes_fluct
        self.emi = emi
        self.lowpass = lowpass
        if Tb is not None:
            self.Tb = Tb
        self.dac_range = dac_range
        self.Ib_range = Ib_range
        self.adc_range = adc_range

        self.t = np.arange(0, record_length / sample_frequency, 1 / sample_frequency)  # s
        self.power_freq = (1e3 + np.abs(np.fft.rfft(np.sin(2 * np.pi * self.t * 50) +
                                                    0.5 * np.sin(2 * np.pi * self.t * 100) +
                                                    0.33 * np.sin(2 * np.pi * self.t * 150)))) ** 2
        self.t0_idx = np.searchsorted(self.t, self.t0)
        self.tpa_idx = 0
        self.timer = 0
        self.T = self.Tb(0) * np.ones((self.record_length, self.nmbr_components))
        self.Il, self.It = self.currents(self.T[:, self.tes_flag])
        self.calc_out()
        self.store_raw = store_raw
        self.pileup_t0 = None
        self.pileup_er = 0

        self.Ce = self.C[self.tes_flag]
        self.update_capacity()

        self.clear_buffer()

        assert self.tp_interval > 2 * self.record_length / self.sample_frequency, \
            'tp_interval must be longer than 2 times the record window'

    # setter and getter

    def norm(self, value, range):  # from range to (-1,1),
        return 2 * (np.array(value) - range[0]) / (range[1] - range[0]) - 1

    def denorm(self, value, range):  # from (-1,1) to range,
        return range[0] + (np.array(value) + 1) / 2 * (range[1] - range[0])

    def set_control(self, dac, Ib, norm=True):
        """
        TODO
        """
        assert len(dac) == self.nmbr_heater, ''
        assert len(Ib) == self.nmbr_tes, ''
        if norm:
            dac = self.denorm(dac, self.dac_range)
            Ib = self.denorm(Ib, self.Ib_range)
        self.dac = np.array(dac)
        self.Ib = np.array(Ib)

        self.update_capacity()

    def get(self, name, norm=False):
        value = np.array(eval('self.' + name))
        if norm:
            if name in ['ph', 'rms', 'offset']:
                value = self.norm(value, self.adc_range)
            if name == 'Ib':
                value = self.norm(value, self.Ib_range)
            else:
                value = self.norm(value, self.dac_range)
        return value

    def get_buffer(self, name):
        """
        TODO
        """
        return np.array(eval('self.buffer_' + name))

    def get_record(self):
        """
        Get the squid output record window.

        :return: The output of the squids, in shape (nmbr_tes, record_length).
        :rtype: numpy array
        """
        return np.array(self.squid_out_noise)

    # public

    def calc_out(self):
        """
        TODO
        """
        self.squid_out = self.xi * (self.Il - self.Il[0])  # remove offset
        self.squid_out[self.squid_out > self.adc_range[1]] = self.adc_range[1]
        self.squid_out[self.squid_out < self.adc_range[0]] = self.adc_range[0]
        self.squid_out_noise = np.copy(self.squid_out)

    def wait(self, seconds, update_T=True):
        """
        TODO
        """
        self.timer += seconds
        self.update_capacitor(seconds)
        if update_T:
            self.pileup_t0 = None
            self.pileup_er = 0
            self.er = np.zeros(self.nmbr_components)
            self.tpa = np.zeros(self.nmbr_heater)
            self.t = np.linspace(0, seconds, self.record_length)

            TIb = odeint(self.dTdItdt,
                         np.concatenate((self.T[-1, :], self.It[-1].reshape(-1))),
                         self.t, args=(
                    self.C, self.Gb, self.Tb, self.G, self.P, self.Rs, self.Ib, self.Rt, self.L, self.tes_flag,
                    self.timer),
                         tfirst=True,
                         )

            self.T = TIb[:, :self.nmbr_components]
            self.It = TIb[:, self.nmbr_components:]
            self.Il = self.Ib - self.It
            self.calc_out()
            self.calc_par()

    def trigger(self, er, tpa, verb=False, store=True, time_passes=True):
        """
        TODO
        """
        assert len(er) == self.nmbr_components, ''
        assert len(tpa) == self.nmbr_heater, ''
        er = np.array(er)
        tpa = np.array(tpa)
        self.er = er
        self.tpa = tpa
        self.t = np.arange(0, self.record_length / self.sample_frequency, 1 / self.sample_frequency)
        if np.random.uniform() < self.pileup_prob:
            self.pileup_t0 = np.random.choice(self.t)
            self.pileup_er = self.pileup_er_distribution()
        else:
            self.pileup_t0 = None
            self.pileup_er = 0
        if verb:
            print(f'T0 is {self.T[-1, :]} mK.')
        tstamp = time.time()

        TIb = odeint(func=self.dTdItdt,
                     y0=np.concatenate((self.T[-1, :], self.It[-1].reshape(-1))),
                     t=self.t,
                     args=(self.C, self.Gb, self.Tb, self.G, self.P, self.Rs, self.Ib, self.Rt, self.L, self.tes_flag,
                           self.timer),
                     tcrit=np.linspace(self.t0, self.t0 + self.record_length / 8 / self.sample_frequency, 10),
                     tfirst=True,
                     )

        self.T = TIb[:, :self.nmbr_components]
        self.It = TIb[:, self.nmbr_components:]
        self.Il = self.Ib - self.It
        self.calc_out()
        if verb:
            print(f'Calculated in {time.time() - tstamp} s.')
            tstamp = time.time()
        for c in np.arange(self.nmbr_tes):
            self.squid_out_noise[:, c] += self.get_noise_bl(tes_channel=c)
        if verb:
            print(f'Generated noise in {time.time() - tstamp} s.')
        self.calc_par()
        if store:
            self.append_buffer()
        if time_passes:
            self.timer += self.t[-1]
            self.update_capacitor(self.t[-1])

    def sweep_dac(self, start, end, heater_channel=0, norm=False):
        """
        TODO
        """
        if norm:
            start = self.denorm(start, self.dac_range)
            end = self.denorm(end, self.dac_range)
        for dac in tqdm(np.arange(start, end, np.sign(end - start) * self.dac_ramping_speed[0] * self.tp_interval)):
            self.dac[heater_channel] = np.array(dac)
            self.wait(seconds=self.tp_interval - self.record_length / self.sample_frequency)
            self.update_capacity()
            self.trigger(er=np.zeros(self.nmbr_components), tpa=self.tpa_queue[self.tpa_idx],
                         verb=False, store=True, time_passes=True)
            self.tpa_idx += 1
            if self.tpa_idx + 1 > len(self.tpa_queue):
                self.tpa_idx = 0

    def sweep_Ib(self, start, end, tes_channel=0, norm=False):
        """
        TODO
        """
        if norm:
            start = self.denorm(start, self.Ib_range)
            end = self.denorm(end, self.Ib_range)
        for Ib in tqdm(np.arange(start, end, np.sign(end - start) * self.Ib_ramping_speed[0] * self.tp_interval)):
            self.Ib[tes_channel] = np.array(Ib)
            self.wait(seconds=self.tp_interval - self.record_length / self.sample_frequency)
            self.update_capacity()
            self.trigger(er=np.zeros(self.nmbr_components), tpa=self.tpa_queue[self.tpa_idx],
                         verb=False, store=True, time_passes=True)
            self.tpa_idx += 1
            if self.tpa_idx + 1 > len(self.tpa_queue):
                self.tpa_idx = 0

    def send_testpulses(self, nmbr_tp=1):
        """
        TODO
        """
        for i in trange(nmbr_tp):
            self.wait(seconds=self.tp_interval - self.record_length / self.sample_frequency)
            self.update_capacity()
            self.trigger(er=np.zeros(self.nmbr_components), tpa=self.tpa_queue[self.tpa_idx],
                         verb=False, store=True, time_passes=True)
            self.tpa_idx += 1
            if self.tpa_idx + 1 > len(self.tpa_queue):
                self.tpa_idx = 0

    def clear_buffer(self):
        """
        TODO
        """
        self.buffer_offset = []
        self.buffer_ph = []
        self.buffer_rms = []
        self.buffer_dac = []
        self.buffer_Ib = []
        self.buffer_tpa = []
        self.buffer_timer = []
        self.buffer_channel = []
        self.buffer_tes_resistance = []
        self.buffer_pulses = []  # TODO buffer for pulses

    def plot_temperatures(self, show=True, save_path=None):

        fig, axes = plt.subplots(self.nmbr_components, 1, figsize=(10, 1.5 * self.nmbr_components), sharex=True)

        power_input = map(self.P, self.t, self.T, self.It)  # for i in range(self.t.shape[0])]
        power_input = np.array(list(power_input))

        for i in range(self.nmbr_components):
            label = 'TES / Component' if self.tes_flag[i] else 'Component'
            axes[i].tick_params(axis='y', labelcolor='C0')
            axes[i].plot(self.t, self.T[:, i], label='Temperature {} {} (mK)'.format(label, i), zorder=10, c='C0',
                         linewidth=2)
            axes[i].legend(loc='upper right', frameon=False).set_zorder(100)
            axes[i].set_zorder(10)
            axes[i].set_frame_on(False)
            axes[i].tick_params(axis='y', labelcolor='C0')
            ax_2 = axes[i].twinx()
            ax_2.plot(self.t, power_input[:, i], label='Heat input {} {} (keV / s)'.format(label, i), c='C3', linewidth=2)
            ax_2.legend(loc='center right', frameon=False).set_zorder(100)
            ax_2.tick_params(axis='y', labelcolor='C3')

        fig.supxlabel('Time (s)')
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            return fig, axes

    def plot_tes(self, show=True, save_path=None):

        fig, axes = plt.subplots(self.nmbr_tes, 2, figsize=(10, 3 * self.nmbr_tes))
        axes = axes.flatten()

        for i in range(self.nmbr_tes):
            tes_channel = i
            t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]  # from tes idx to component idx

            # transition curve plot
            Tmin, Tmax = np.min(self.T[:, t_ch]), np.max(self.T[:, t_ch])
            Rmin, Rmax = self.Rt[tes_channel](Tmin), self.Rt[tes_channel](Tmax)
            temp = np.linspace(np.minimum(self.Tc[tes_channel] - 4 / self.k[tes_channel], np.minimum(Tmin, self.Tb(0))),
                               np.maximum(self.Tc[tes_channel] + 4 / self.k[tes_channel], Tmax), 100)
            axes[2*i + 0].plot(temp, 1000 * self.Rt[tes_channel](temp), label='Transition curve', c='#FF7979', linewidth=2)
            axes[2*i + 0].axvline(x=self.Tb(0), color='grey', linestyle='dashed', label='Heat bath')
            axes[2*i + 0].fill_between([Tmin, Tmax],
                                    [0, 0],
                                    [1000 * Rmin, 1000 * Rmax], color='#99CCFF', alpha=0.5, zorder=10)
            axes[2*i + 0].fill_between([temp[0], Tmin, Tmax],
                                    [1000 * Rmin, 1000 * Rmin, 1000 * Rmax],
                                    [1000 * Rmax, 1000 * Rmax, 1000 * Rmax], color='#99CCFF', alpha=0.5, zorder=10)
            axes[2*i + 0].plot([temp[0], Tmin], [1000 * Rmin, 1000 * Rmin], color='black',  # alpha=0.5,
                            linewidth=2, label='OP', zorder=100)
            axes[2*i + 0].plot([temp[0], Tmax], [1000 * Rmax, 1000 * Rmax], color='black',  # alpha=0.5,
                            linewidth=2, zorder=100)
            axes[2*i + 0].plot([Tmin, Tmin], [0, 1000 * Rmin], color='black',  # alpha=O.5,
                            linewidth=2, zorder=100)
            axes[2*i + 0].plot([Tmax, Tmax], [0, 1000 * Rmax], color='black',  # alpha=0.5,
                            linewidth=2, zorder=100)
            axes[2*i + 0].set_ylabel('Resistance (mOhm)', c='#FF7979')
            axes[2*i + 0].set_xlabel('Temperature (mK)')
            axes[2*i + 0].legend(frameon=False).set_zorder(100)
            axes[2*i + 0].tick_params(axis='y', labelcolor='#FF7979')
            axes[2*i + 0].set_title('TES curve {}'.format(i))

            # recoil signature plot
            axes[2*i + 1].plot(self.t, self.squid_out_noise[:, i], label='Squid output', zorder=5, c='black', linewidth=1,
                            alpha=0.7)
            axes[2*i + 1].plot(self.t, self.squid_out[:, i], label='Recoil signature', zorder=10, c='red', linewidth=2,
                            alpha=1)
            axes[2*i + 1].set_ylabel('Voltage (V)')  # , color='red'
            axes[2*i + 1].set_xlabel('Time (s)')
            axes[2*i + 1].legend(loc='upper right', frameon=False).set_zorder(100)
            axes[2*i + 1].set_zorder(10)
            axes[2*i + 1].set_title('Squid output {}'.format(i))

        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            return fig, axes

    def plot_event(self, tes_channel=0, show=True):
        """
        TODO
        deprecated!!
        """

        assert self.nmbr_components == 2, 'This method works only with the standard 2 component design!'
        assert self.nmbr_tes == 1, 'This method works only with the standard 2 component design!'

        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]  # from tes idx to component idx

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))

        # temperatures plot
        axes[0, 0].tick_params(axis='y', labelcolor='C0')
        axes[0, 0].plot(self.t, self.T[:, 0], label='Thermometer', zorder=10, c='C0', linewidth=2)
        axes[0, 0].set_ylabel('Temperature (mK)', color='C0')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].legend(loc='upper right').set_zorder(100)
        axes[0, 0].set_zorder(10)
        axes[0, 0].set_frame_on(False)
        axes[0, 0].tick_params(axis='y', labelcolor='C0')
        ax00_2 = axes[0, 0].twinx()
        ax00_2.plot(self.t, self.T[:, 1], label='Crystal', c='C1', linewidth=2)
        ax00_2.set_ylabel('Temperature (mK)', color='C1')
        ax00_2.legend(loc='center right').set_zorder(100)
        ax00_2.tick_params(axis='y', labelcolor='C1')

        # recoil signature plot
        axes[0, 1].plot(self.t, self.squid_out_noise, label='Squid output', zorder=5, c='black', linewidth=1, alpha=0.7)
        axes[0, 1].plot(self.t, self.squid_out, label='Recoil signature', zorder=10, c='red', linewidth=2, alpha=1)
        axes[0, 1].set_ylabel('Voltage (V)')  # , color='red'
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].legend(loc='upper right').set_zorder(100)
        axes[0, 1].set_zorder(10)

        # transition curve plot
        Tmin, Tmax = np.min(self.T[:, t_ch]), np.max(self.T[:, t_ch])
        Rmin, Rmax = self.Rt[tes_channel](Tmin), self.Rt[tes_channel](Tmax)
        temp = np.linspace(np.minimum(self.Tc[tes_channel] - 4 / self.k[tes_channel], np.minimum(Tmin, self.Tb(0))),
                           np.maximum(self.Tc[tes_channel] + 4 / self.k[tes_channel], Tmax), 100)
        axes[1, 0].plot(temp, 1000 * self.Rt[tes_channel](temp), label='Transition curve', c='#FF7979', linewidth=2)
        axes[1, 0].axvline(x=self.Tb(0), color='grey', linestyle='dashed', label='Heat bath')
        axes[1, 0].fill_between([Tmin, Tmax],
                                [0, 0],
                                [1000 * Rmin, 1000 * Rmax], color='#99CCFF', alpha=0.5, zorder=10)
        axes[1, 0].fill_between([temp[0], Tmin, Tmax],
                                [1000 * Rmin, 1000 * Rmin, 1000 * Rmax],
                                [1000 * Rmax, 1000 * Rmax, 1000 * Rmax], color='#99CCFF', alpha=0.5, zorder=10)
        axes[1, 0].plot([temp[0], Tmin], [1000 * Rmin, 1000 * Rmin], color='black',  # alpha=0.5,
                        linewidth=2, label='OP', zorder=100)
        axes[1, 0].plot([temp[0], Tmax], [1000 * Rmax, 1000 * Rmax], color='black',  # alpha=0.5,
                        linewidth=2, zorder=100)
        axes[1, 0].plot([Tmin, Tmin], [0, 1000 * Rmin], color='black',  # alpha=O.5,
                        linewidth=2, zorder=100)
        axes[1, 0].plot([Tmax, Tmax], [0, 1000 * Rmax], color='black',  # alpha=0.5,
                        linewidth=2, zorder=100)
        axes[1, 0].set_ylabel('Resistance (mOhm)', c='#FF7979')
        axes[1, 0].set_xlabel('Temperature (mK)')
        axes[1, 0].legend().set_zorder(100)
        axes[1, 0].tick_params(axis='y', labelcolor='#FF7979')

        # power input plot
        power_input = map(self.P, self.t, self.T, self.It)  # for i in range(self.t.shape[0])]
        power_input = np.array(list(power_input))
        axes[1, 1].plot(self.t, 0.001 * power_input[:, 0], color='blue', label='Thermometer', zorder=10, linewidth=2,
                        alpha=0.6)
        axes[1, 1].set_ylabel('Heat input (MeV / s)', color='blue')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].legend().set_zorder(100)
        axes[1, 1].set_zorder(10)
        axes[1, 1].set_frame_on(False)
        axes[1, 1].tick_params(axis='y', labelcolor='blue')
        ax11_2 = axes[1, 1].twinx()
        ax11_2.plot(self.t, 0.001 * power_input[:, 1], color='orange', label='Crystal', linewidth=2)
        ax11_2.set_ylabel('Heat input (MeV / s)', color='orange')
        ax11_2.legend(loc='center right').set_zorder(100)
        ax11_2.tick_params(axis='y', labelcolor='orange')

        fig.tight_layout()
        if show:
            plt.show()
        else:
            return fig, axes

    def plot_nps(self, tes_channel=0, only_sum=False, save_path=None):
        """
        TODO
        """

        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        Tt = self.T[0, t_ch]
        It = self.It[0, t_ch]

        fig, ax = plt.subplots(1, 1, figsize=(10, 3*self.nmbr_tes))

        w, nps = self.get_nps(Tt, It, tes_channel)
        ax.loglog(w, 1e6 * np.sqrt(nps), label='combined', linewidth=2, color='black', zorder=10)

        if not only_sum:
            w = np.logspace(np.log10(np.min(w[w > 0])), np.log10(np.max(w)))
            nps = self.thermal_noise(w, Tt, It, tes_channel)
            ax.loglog(w, 1e6 * np.sqrt(nps), label='Thermal noise', linewidth=2)

            nps[w > 0] = self.thermometer_johnson(w, Tt, It, tes_channel)
            ax.loglog(w, 1e6 * np.sqrt(nps), label='Thermometer Johnson', linewidth=2)

            nps = self.shunt_johnson(w, Tt, It, tes_channel)
            ax.loglog(w, 1e6 * np.sqrt(nps), label='Shunt Johnson', linewidth=2)

            nps = self.squid_noise(w, Tt, It, tes_channel)
            ax.loglog(w, 1e6 * np.sqrt(nps), label='Squid noise', linewidth=2)

            nps = self.one_f_noise(w, Tt, It, tes_channel)
            ax.loglog(w, 1e6 * np.sqrt(nps), label='1/f noise', linewidth=2)

            nps = self.emi_noise(w[w > 0], Tt, It, tes_channel)
            ax.loglog(w[w > 0], 1e6 * np.sqrt(nps), label='EM interference', linewidth=2)

        ax.set_title('Squid output {}'.format(tes_channel))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (pA / sqrt(Hz))')
        ax.legend(frameon=False, bbox_to_anchor=(1., 1.))

        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def print_noise_parameters(self, channel=0):  # its the TES channel
        """
        TODO
        """

        t_ch = np.arange(self.nmbr_components)[self.tes_flag][channel]
        Tt = self.T[0, t_ch]
        It = self.It[0, t_ch]

        print('Resistance TES / Resistance normal conducting: {}'.format(self.Rt[channel](Tt) / self.Rt0[channel]))
        print('Temperature mixing chamber: {} mK'.format(self.Tb(0)))
        print('Temperature TES: {} mK'.format(Tt))
        print('Resistance TES: {} mOhm'.format(1e3 * self.Rt[channel](Tt)))
        print('Tau eff: {} ms'.format(1e3 * self.tau_eff(Tt, It, channel)))
        print('Slope: {} mOhm/mK'.format(1e3 * self.dRtdT(Tt, channel)))
        print('C: {} fJ / K '.format(1e6 * self.C[t_ch]))  # should be Pantic 3.5, and cubic
        print('Geff: ???')
        print('Tau in: {} ms'.format(1e3 * self.tau_in(channel)))
        print('Geb: {} pW / K '.format(1e3 * self.Gb[0]))
        print('G ETF: {} pW / K '.format(1e3 * self.G_etf(Tt, It, channel)))
        print('R shunt: {} mOhm'.format(1e3 * self.Rs[channel]))
        print('Temperature shunt: {} mK'.format(self.Tb(0)))
        print('i sq: {} pA/sqrt(Hz)'.format(1e12 * self.i_sq[channel]))
        print('1 / f amplitude: {} '.format(self.tes_fluct[channel] ** 2))

    def plot_buffer(self, tes_channel=0, tpa=None, save_path=None):
        """
        TODO
        """

        buffer_channel = self.get_buffer('channel')
        buffer_Ib = self.get_buffer('Ib')[buffer_channel == tes_channel]
        buffer_dac = self.get_buffer('dac')[buffer_channel == tes_channel]
        buffer_ph = self.get_buffer('ph')[buffer_channel == tes_channel]
        buffer_timer = self.get_buffer('timer')[buffer_channel == tes_channel]
        buffer_tpa = self.get_buffer('tpa')[buffer_channel == tes_channel]

        if tpa is not None:
            buffer_Ib = buffer_Ib[buffer_tpa == tpa]
            buffer_dac = buffer_dac[buffer_tpa == tpa]
            buffer_ph = buffer_ph[buffer_tpa == tpa]
            buffer_timer = buffer_timer[buffer_tpa == tpa]

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))

        axes[0, 0].scatter(buffer_dac, buffer_ph,
                           s=5, marker='o', edgecolor='black', color='white')
        axes[0, 0].set_xlabel('DAC (V)')
        axes[0, 0].set_ylabel('PH (V)')

        axes[0, 1].scatter(buffer_timer, buffer_ph,
                           s=5, marker='o', edgecolor='black', color='white')
        axes[0, 1].set_xlabel('Timer (s)')
        axes[0, 1].set_ylabel('PH (V)')

        axes[1, 0].scatter(buffer_Ib, buffer_ph,
                           s=5, marker='o', edgecolor='black', color='white')
        axes[1, 0].set_xlabel('Ib (muA)')
        axes[1, 0].set_ylabel('PH (V)')

        colors_ph = (buffer_ph - np.min(buffer_ph)) / np.max(buffer_ph)
        reds = plt.get_cmap('Reds')

        axes[1, 1].scatter(buffer_Ib, buffer_dac,
                           s=10, marker='o', color=reds(colors_ph))
        axes[1, 1].set_xlabel('Ib (muA)')
        axes[1, 1].set_ylabel('DAC (V)')

        plt.suptitle('Buffer channel {}'.format(tes_channel))
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

    def write_buffer(self, path):
        """
        TODO
        """

        buffer_Ib = self.get_buffer('Ib')
        buffer_dac = self.get_buffer('dac')
        buffer_ph = self.get_buffer('ph')
        buffer_timer = self.get_buffer('timer')
        buffer_tpa = self.get_buffer('tpa')
        buffer_rms = self.get_buffer('rms')
        buffer_tes_resistance = self.get_buffer('tes_resistance')
        buffer_pulses = self.get_buffer('pulses')

        df = {
            'Channel': np.ones(buffer_Ib.shape),
            'Time (h)': buffer_timer / 3600,
            'Pulse height (V)': buffer_ph,
            'RMS (V)': buffer_rms,
            'Test pulse amplitude (V)': buffer_tpa,
            'DAC output (V)': buffer_dac,
            'Bias current (muA)': buffer_Ib,
            'TES resistance / normal conducting': buffer_tes_resistance,
        }

        df = pd.DataFrame(df)

        # store csv file
        df.to_csv(path + '.csv')

        if self.store_raw:
            np.save(path + '.npy', buffer_pulses, )

    # time dependent variables

    @staticmethod
    def Tb(t):
        """
         The time-dependent heat bath temperature.

        :param t: The timer time in seconds (e.g. self.timer).
        :type t: float
        :return: The heat bath temperature in ms.
        :rtype: float
        """
        T = 11.  # temp bath
        return T

    def update_capacitor(self, delta_t):
        self.U_sq_Rh = (self.U_sq_Rh - self.dac) * np.exp(- delta_t / self.tau) + self.dac

    def P(self, t, T, It, no_pulses=False):
        """
        TODO
        """
        keV_to_pJ = e * 1e3 * 1e12
        P = np.zeros(T.shape)
        if t > self.t0 and not no_pulses:
            for i in range(self.nmbr_components):
                P += self.er[i] * self.eps[i] * np.exp(
                    -(t - self.t0) / self.lamb) / self.lamb * keV_to_pJ  # particle
        if self.pileup_t0 is not None and t > self.pileup_t0 and not no_pulses:
            P += self.pileup_er * self.eps[self.pileup_comp] * np.exp(
                -(t - self.pileup_t0) / self.lamb) / self.lamb * keV_to_pJ  # pile up particle
        for i in range(self.nmbr_tes):
            c = np.nonzero(self.tes_flag)[0][i]
            P[c] += self.Rt[i](T[c]) * It[i] ** 2  # self heating
        for i in range(self.nmbr_heater):
            P += self.delta[i] * self.heater_attenuator[i] * self.U_sq_Rh[i] / self.Rh[i]  # heating
        if t > self.t0 and not no_pulses:
            for i in range(self.nmbr_heater):
                P += np.maximum(self.tpa[i], 0) * self.delta[i] * self.heater_attenuator[i] * self.pulser_scale[
                    i] * np.exp(
                    -(t - self.t0) / self.lamb) / self.Rh[i]  # test pulses
        return P

    # private

    @staticmethod
    def dTdItdt(t, TIt, C, Gb, Tb, G, P, Rs, Ib, Rt, L, tes_flag, timer):
        """
        TODO
        """
        nmbr_components = C.shape[0]
        dTdIt = np.zeros(nmbr_components + Ib.shape[0])
        T = TIt[:nmbr_components]
        It = TIt[nmbr_components:]

        # thermal
        dTdIt[:nmbr_components] = P(t, T, It)  # heat input
        dTdIt[:nmbr_components] += Gb * (Tb(t + timer) - T)  # coupling to temperature bath
        dTdIt[:nmbr_components] += np.dot(G, T)  # heat transfer from other components
        dTdIt[:nmbr_components] -= np.dot(np.diag(np.dot(G, np.ones(T.shape[0]))),
                                          T)  # heat transfer to other components
        dTdIt[:nmbr_components] /= C  # heat to temperature

        # electricalv
        dTdIt[nmbr_components:] = Rs * Ib  #
        for i in range(Ib.shape[0]):
            c = np.nonzero(tes_flag)[0][i]
            dTdIt[nmbr_components + i] -= It[i] * (Rt[i](T[c]) + Rs[i])  #
        dTdIt[nmbr_components:] /= L  # voltage to current

        return dTdIt

    def Rt_init(self, which_curve, k, Tc, Rt0):
        """
        TODO
        """
        return eval(which_curve)(k, Tc, Rt0)

    @staticmethod
    def pileup_er_distribution():
        """
        Recoil energy distribution of the pile up events.

        :return: A recoil energy for one pile-up event, in keV.
        :rtype: float
        """
        return np.random.exponential(10)  # in keV

    def currents(self, Tt):
        """
        TODO
        Tt has shape (record_length, nmbr_tes)
        """
        Rs = self.Rs
        R_tes, Il, It = np.zeros(Tt.shape), np.zeros(Tt.shape), np.zeros(Tt.shape)
        for i in range(self.nmbr_tes):
            R_tes[:, i] = self.Rt[i](Tt[:, i])
            Il[R_tes[:, i] > 0, i] = self.Ib[i] * (1 / (1 + Rs[i] / R_tes[R_tes[:, i] > 0, i]))
            It[R_tes[:, i] > 0, i] = self.Ib[i] * (1 / (1 + R_tes[R_tes[:, i] > 0, i] / Rs[i]))
            It[R_tes[:, i] <= 0, i] = self.Ib[i]
        return Il, It

    def update_capacity(self):
        """
        TODO
        I think this gives some unexpected behavior for triggering twice without waiting
        """
        for i in range(self.nmbr_tes):
            tes_idx = np.nonzero(self.tes_flag)[0][i]
            self.C[tes_idx] = self.Ce[i] * (2.43 - 1.43 * self.Rt[i](self.T[0, tes_idx]) / self.Rt0[i])

    # noise contributions

    def get_noise_bl(self, tes_channel=0):
        """
        Get a simulated noise baseline that can be superposed with the squid output.

        :param channel: The number of the TES for which the noise is simulated.
        :type channel: int
        :return: The noise contribution to the squid output in V.
        :rtype: 1D numpy array
        """

        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        Tt = self.T[0, t_ch]  # in mK
        It = self.It[0, t_ch]  # in muA

        w, nps = self.get_nps(Tt, It, tes_channel)  # nps in (muA)^2/Hz

        noise = self.noise_function(self.record_length * nps, size=1)[0] * (1 + 0.01 * np.random.normal())  # in muA

        return self.xi[tes_channel] * noise  # in V

    @staticmethod
    def noise_function(nps, size):
        """
        A value trace in time space, that follows the given NPS.

        :param nps: The noise power spectrum in (muA)^2/Hz.
        :type nps: 1D numpy array
        :param size: The number of traces to simulate.
        :type size: int
        :return: The simulated noise trace in muA.
        :rtype: 1D numpy array
        """
        f = np.sqrt(nps)
        f = np.array(f, dtype='complex')  # create array for frequencies
        f = np.tile(f, (size, 1))
        phases = np.random.rand(size, nps.shape[0]) * 2 * np.pi  # create random phases
        phases = np.cos(phases) + 1j * np.sin(phases)
        f *= phases
        return np.fft.irfft(f, axis=-1)

    def get_nps(self, Tt, It, tes_channel=0):
        """
        Simulate the total noise power spectrum with all contributions.

        :param Tt: The temperature of the corresponding TES, in mK.
        :type Tt: float
        :param It: The current through the corresponsing TES, in muA.
        :type It: float
        :param channel: The number of the TES.
        :type channel: int
        :return: The frequencies of the nps (Hz) and their amplitudes in (muA)^2/Hz.
        :rtype: list of two 1D numpy arrays
        """
        w = np.fft.rfftfreq(self.record_length, 1 / self.sample_frequency)
        nps = np.zeros(w.shape)
        nps[w > 0] = self.thermal_noise(w[w > 0], Tt, It, tes_channel) + \
                     self.thermometer_johnson(w[w > 0], Tt, It, tes_channel) + \
                     self.shunt_johnson(w[w > 0], Tt, It, tes_channel) + \
                     self.squid_noise(w[w > 0], Tt, It, tes_channel) + \
                     self.one_f_noise(w[w > 0], Tt, It, tes_channel) + \
                     self.emi_noise(w[w > 0], Tt, It, tes_channel)
        if self.lowpass is not None:
            b, a = butter(N=1, Wn=self.lowpass, btype='lowpass', analog=True)
            _, h = freqs(b, a, worN=w[w > 0])
            nps[w > 0] *= np.abs(h)
        return w, nps

    def dRtdT(self, T, tes_channel=0):
        """
        TODO
        """
        grid = np.linspace(self.Tc[tes_channel] - 5 / self.k[tes_channel],
                           self.Tc[tes_channel] + 5 / self.k[tes_channel], 1000)
        h = (grid[1] - grid[0])
        return np.interp(T, grid[:-1] + h / 2, 1 / h * np.diff(self.Rt[tes_channel](grid)))

    def G_etf(self, Tt, It, tes_channel=0):
        """
        TODO
        """
        return It ** 2 * self.dRtdT(Tt, tes_channel) * (self.Rt[tes_channel](Tt) - self.Rs[tes_channel]) / (
                    self.Rt[tes_channel](Tt) + self.Rs[tes_channel])

    def tau_eff(self, Tt, It, tes_channel=0):
        """
        TODO
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        return self.tau_in(tes_channel) / (1 + self.G_etf(Tt, It) / self.Gb[t_ch])

    def tau_in(self, tes_channel=0):
        """
        TODO
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        return self.C[t_ch] / self.Gb[t_ch]

    def thermal_noise(self, w, Tt, It, tes_channel=0):
        """
        TODO
        return in (muA)^2/Hz
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        P_th_squared = 4 * k * Tt ** 2 * self.Gb[t_ch]
        if Tt > self.Tb(0):
            P_th_squared *= 2 / 5 * (1 - (self.Tb(0) / Tt) ** 5) / (1 - (self.Tb(0) / Tt) ** 2)
        S_squared = 1 / (1 + w ** 2 * self.tau_eff(Tt, It, tes_channel) ** 2)
        S_squared /= (self.Gb[t_ch] + self.G_etf(Tt, It, tes_channel)) ** 2
        S_squared *= (It / (self.Rt[tes_channel](Tt) + self.Rs[tes_channel])) ** 2
        S_squared *= self.dRtdT(Tt, tes_channel) ** 2
        return S_squared * P_th_squared * 1e6

    def thermometer_johnson(self, w, Tt, It, tes_channel=0):
        """
        TODO
        return in (muA)^2/Hz
        """
        I_jt = (1 + w ** 2 * self.tau_in(tes_channel) ** 2)
        I_jt /= (1 + w ** 2 * self.tau_eff(Tt, It, tes_channel) ** 2)
        I_jt *= 4 * k * Tt * self.Rt[tes_channel](Tt) / (self.Rt[tes_channel](Tt) + self.Rs[tes_channel]) ** 2
        I_jt *= (self.tau_eff(Tt, It, tes_channel) / self.tau_in(tes_channel)) ** 2
        return I_jt * 1e9

    def shunt_johnson(self, w, Tt, It, tes_channel=0):
        """
        TODO
        return in (muA)^2/Hz
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        I_js = (1 - It ** 2 / self.Gb[t_ch] * self.dRtdT(Tt, tes_channel)) ** 2 + w ** 2 * self.tau_in(tes_channel) ** 2
        I_js /= (1 + w ** 2 * self.tau_eff(Tt, It, tes_channel) ** 2)
        I_js *= 4 * k * self.Tb(self.timer) * self.Rs[tes_channel] / (
                    self.Rt[tes_channel](Tt) + self.Rs[tes_channel]) ** 2
        I_js *= (self.tau_eff(Tt, It, tes_channel) / self.tau_in(tes_channel)) ** 2
        return I_js * 1e9

    def squid_noise(self, w, Tt, It, tes_channel=0):
        """
        TODO
        return in (muA)^2/Hz
        """
        I_sq = w / w
        I_sq *= self.i_sq[tes_channel] ** 2
        return I_sq * 1e12

    def one_f_noise(self, w, Tt, It, tes_channel=0):
        """
        TODO
        return in (muA)^2/Hz
        """
        I_one_f = (1 + w ** 2 * self.tau_in(tes_channel) ** 2)
        I_one_f /= (1 + w ** 2 * self.tau_eff(Tt, It, tes_channel) ** 2)
        I_one_f *= (self.tau_eff(Tt, It, tes_channel) / self.tau_in(tes_channel)) ** 2
        I_one_f *= It ** 2 * self.Rt[tes_channel](Tt) ** 2 * self.tes_fluct[tes_channel] ** 2 / w / (
                    self.Rt[tes_channel](Tt) + self.Rs[tes_channel]) ** 2
        return I_one_f

    def emi_noise(self, w, Tt, It, tes_channel=0):
        """
        TODO
        return in (muA)^2/Hz
        """
        # pdb.set_trace()
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        I_em = (1 - It ** 2 / self.Gb[t_ch] * self.dRtdT(Tt, tes_channel)) ** 2 + w ** 2 * self.tau_in(tes_channel) ** 2
        I_em /= (1 + w ** 2 * self.tau_eff(Tt, It, tes_channel) ** 2)
        I_em /= (self.Rt[tes_channel](Tt) + self.Rs[tes_channel]) ** 2
        I_em *= (self.tau_eff(Tt, It, tes_channel) / self.tau_in(tes_channel)) ** 2
        I_em *= (self.emi[tes_channel]) ** 2 * self.power_freq[-w.shape[0]:]
        return I_em

    # utils

    def calc_par(self):
        """
        TODO
        """
        self.offset = np.mean(self.squid_out_noise[self.t < self.t0], axis=0)
        self.ph = np.max(self.squid_out_noise[self.t >= self.t0] - self.offset, axis=0)
        self.rms = np.std(self.squid_out_noise[self.t < self.t0] - self.offset, axis=0)

    def append_buffer(self):
        """
        TODO
        TODO Attention, the dac and tes channels are wrongly aligned! (problem for more components)
        """
        self.buffer_offset.extend(self.offset)  # scalar
        self.buffer_ph.extend(self.ph)  # scalar
        self.buffer_rms.extend(self.rms)  # scalar
        self.buffer_dac.extend(self.dac)
        self.buffer_Ib.extend(self.Ib)
        self.buffer_tpa.extend(self.tpa * np.ones(self.nmbr_tes))  # scalar
        self.buffer_timer.extend(self.timer * np.ones(self.nmbr_tes))  # scalar
        self.buffer_channel.extend(np.arange(self.nmbr_tes))
        self.buffer_tes_resistance.extend(self.Rt[0](self.T[0, 0]) / self.Rt0[0] * np.ones(
            self.nmbr_tes))  # TODO buffer not okay for multiple channels!
        if self.store_raw:
            pulse = self.squid_out_noise.reshape(256, -1)  # down sample for storage
            pulse = np.mean(pulse, axis=-1)
            self.buffer_pulses.append(pulse)

        for ls in [self.buffer_offset, self.buffer_ph, self.buffer_rms,
                   self.buffer_dac, self.buffer_Ib, self.buffer_tpa,
                   self.buffer_timer, self.buffer_channel, self.buffer_tes_resistance,
                   self.buffer_pulses]:
            while len(ls) > self.max_buffer_len:
                del ls[0]
