import pdb

import numpy as np
from scipy.integrate import odeint
from scipy.constants import e, k
from scipy.signal import butter, freqs
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm, trange
import pandas as pd
from ._transition_curves import Rt_smooth, Rt_kinky
from ._heat_capacities import tes_heat_capacity
from scipy.optimize import fsolve, minimize, brute, fmin, brentq
from copy import deepcopy


class DetectorModel:
    """
    TODO
    """

    kB = 1.380649e-17  # mm ^ 2 * g / s^2 / mK     # nJ / mK
    na = 6.02214076e23  # number of items per mole
    h_const = 6.62607015e-34  # kg * m^2 / s
    e_charge = 1.60217663e-19  # coulombs
    keV_to_pJ = e * 1e3 * 1e12

    def __init__(self,
                 record_length=16384,
                 sample_frequency=25000,
                 C=None,  # pJ / mK, in normal conducting state, is defined later bec mutable
                 Gb=np.array([1.2e-1, 1.5e0]),  # pW / mK
                 G=np.array([[0., 2.14e-1],
                             [2.14e-1, 0.], ]),  # heat cond between components, pW / mK
                 lamb=np.array([0.000382, 0.000382, ]),  # rise time particles (s)
                 lamb_tp=np.array([0.00496]),  # rise time tp (s)
                 eps=np.array([[0.99, 0.01], [0.1384, 1 - 0.1384], ]),
                 # share thermalization in components
                 delta=np.array([[0., 1.], ]),
                 delta_h=np.array([[0., 1.], ]),
                 # share thermalization/heating in components
                 Rs=np.array([0.04]),  # Ohm
                 Rh=np.array([10.]),  # Ohm
                 L=np.array([3.5e-7]),  # H
                 Rt0=np.array([0.11]),  # Ohm
                 k=np.array([4.]),  # 1/mK
                 Tc=np.array([31.]),  # mK
                 Ib=np.array([5]),  # muA,
                 dac=np.array([1.662]),  # V, heating power is (DAC/10) * heater_current ** 2 * Rh
                 pulser_scale=np.array([.073]),  # pulses have power (TPA/10) * pulser_scale * heater_current ** 2 * Rh
                 heater_current=np.array([4.8]),  # muA
                 tes_flag=np.array([True, False], dtype=bool),  # which component is a tes
                 heater_flag=np.array([False, True], dtype=bool),  # which component has a heater
                 t0=.16384,  # onset of the trigger, s
                 pileup_prob=0.00,  # percent / record window
                 pileup_comp=1,
                 tpa_queue=np.array([0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ]),  # V
                 tp_interval=5,  # s
                 max_buffer_len=10000,
                 dac_ramping_speed=np.array([2e-3]),  # V / s
                 Ib_ramping_speed=np.array([5e-3]),  # muA / s
                 eta=np.array([1. / 0.173]),  # squid conversion current to voltage, V / muA
                 i_sq=np.array([1.2]),  # squid noise, muA / sqrt(Hz)
                 tes_fluct=np.array([1.2e-4]),  # pJ
                 flicker_slope=np.array([1.5]),  # negative power of flicker noise
                 emi=np.array([[1.5e-5, 1.e-5, 1.5e-5]]),  # pV
                 lowpass=3.5e3,  # Hz
                 Tb=None,  # function that takes one positional argument t, returns Tb
                 Rt=None,
                 # function that takes one positional argument T, returns Rt
                 which_curve='Rt_smooth',
                 tau_cap=np.array([1.]),
                 dac_range=(0., 1e1),  # V
                 Ib_range=(0, 17.86),  # muA
                 adc_range=(-10., 10.),  # V
                 store_raw=True,
                 excess_phonon=np.array([1.]),
                 excess_johnson=np.array([6.]),
                 verb=True,
                 **kwargs,
                 ):

        if C is None:
            self.C = np.array([1.34e-3, 1.11e-1])
        else:
            self.C = deepcopy(C)

        if Rt is not None:
            self.Rt = deepcopy(Rt)
        else:
            self.Rt = [self.Rt_init(which_curve, k_, Tc_, Rt0_) for k_, Tc_, Rt0_ in zip(k, Tc, Rt0)]

        self.heat_capacity_facts = [tes_heat_capacity(k_, Tc_, Rt_) for k_, Tc_, Rt_ in zip(k, Tc, self.Rt)]

        tpa_queue = np.array(deepcopy(tpa_queue))

        # define number of thermal components
        self.nmbr_components = len(self.C)
        assert len(tes_flag) == self.nmbr_components, ''
        assert len(heater_flag) == self.nmbr_components, ''
        assert len(self.C) == self.nmbr_components, ''
        assert len(G) == self.nmbr_components, ''
        assert len(G[0]) == self.nmbr_components, ''
        assert len(Gb) == self.nmbr_components, ''
        assert pileup_comp < self.nmbr_components, ''

        # define number of tes
        self.nmbr_tes = len(Rs)
        assert len(Rs) == self.nmbr_tes, ''
        assert np.sum(tes_flag) == self.nmbr_tes, ''
        assert len(L) == self.nmbr_tes, ''
        assert len(Ib_ramping_speed) == self.nmbr_tes, ''
        assert len(i_sq) == self.nmbr_tes, ''
        assert len(tes_fluct) == self.nmbr_tes, ''
        assert len(flicker_slope) == self.nmbr_tes, ''
        assert len(excess_phonon) == self.nmbr_tes, ''
        assert len(excess_johnson) == self.nmbr_tes, ''
        assert len(emi) == self.nmbr_tes, ''
        for i in range(self.nmbr_tes):
            assert len(emi[i]) == 3, '{i}.format(i)'
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
            assert Rt[0][np.array([0.01, 0.02])].shape == (2,), ''

        # define number of heaters
        self.nmbr_heater = len(Rh)
        assert len(Rh) == self.nmbr_heater, ''
        assert np.sum(heater_flag) == self.nmbr_heater, ''
        assert len(pulser_scale) == self.nmbr_heater, ''
        assert len(heater_current) == self.nmbr_heater, ''
        assert len(dac) == self.nmbr_heater, ''
        assert len(dac_ramping_speed) == self.nmbr_heater, ''
        assert len(tau_cap) == self.nmbr_heater, ''
        assert len(tpa_queue.shape) == 1 or (len(tpa_queue.shape) == 2 and tpa_queue.shape[1] == self.nmbr_heater), ''

        # thermalization parameters
        assert eps.shape[0] == eps.shape[1] == self.nmbr_components, ''
        assert delta.shape[0] == self.nmbr_heater, ''
        assert delta.shape[1] == self.nmbr_components, ''
        assert delta_h.shape[0] == self.nmbr_heater, ''
        assert delta_h.shape[1] == self.nmbr_components, ''
        assert lamb.shape[0] == self.nmbr_components, ''
        assert lamb_tp.shape[0] == self.nmbr_heater, ''

        if len(tpa_queue.shape) == 1:
            tpa_queue = np.repeat(tpa_queue.reshape(-1, 1), self.nmbr_heater, axis=1)

        self.Gb = deepcopy(Gb)
        self.G = deepcopy(G)
        self.lamb = lamb
        self.lamb_tp = lamb_tp
        self.eps = deepcopy(eps)
        self.delta = deepcopy(delta)
        self.delta_h = deepcopy(delta_h)
        self.Rs = deepcopy(Rs)
        self.Rh = deepcopy(Rh)
        self.Rt0 = deepcopy(Rt0)
        self.L = deepcopy(L)
        self.k = deepcopy(k)
        self.Tc = deepcopy(Tc)
        self.Ib = deepcopy(Ib)
        # self.Ib_eff = np.copy(Ib)
        self.dac = deepcopy(dac)
        self.U_sq_Rh = np.copy(dac)
        self.tau_cap = deepcopy(tau_cap)
        self.pulser_scale = deepcopy(pulser_scale)
        self.heater_current = deepcopy(heater_current)
        self.tes_flag = deepcopy(tes_flag)
        self.heater_flag = deepcopy(heater_flag)
        self.record_length = deepcopy(record_length)
        self.sample_frequency = deepcopy(sample_frequency)
        self.t0 = t0
        self.pileup_prob = pileup_prob
        self.pileup_comp = pileup_comp
        self.tpa_queue = deepcopy(tpa_queue)
        self.tp_interval = tp_interval
        self.max_buffer_len = max_buffer_len
        self.dac_ramping_speed = deepcopy(dac_ramping_speed)
        self.Ib_ramping_speed = deepcopy(Ib_ramping_speed)
        self.eta = deepcopy(eta)
        self.i_sq = deepcopy(i_sq)
        self.tes_fluct = deepcopy(tes_fluct)
        self.emi = deepcopy(emi)
        self.lowpass = lowpass
        if Tb is not None:
            self.Tb = Tb
        self.dac_range = dac_range
        self.Ib_range = Ib_range
        self.adc_range = adc_range
        self.flicker_slope = deepcopy(flicker_slope)
        self.excess_phonon = deepcopy(excess_phonon)
        self.excess_johnson = deepcopy(excess_johnson)

        self.t = np.arange(0, record_length / sample_frequency, 1 / sample_frequency)  # s
        self.t0_idx = np.searchsorted(self.t, self.t0)
        self.tpa_idx = 0
        self.timer = 0
        self.T = self.Tb(0) * np.ones((self.record_length, self.nmbr_components))
        self.Il, self.It = self.currents(self.T[:, self.tes_flag])
        self.calc_out()
        self.store_raw = store_raw
        self.pileup_t0 = None
        self.pileup_er = 0

        self.verb = verb

        self.kwargs = kwargs

        # create the normalization factors for RFFT (use norm=backward) and IRFFT (use norm=forward)
        # they are to be applied on the squared noise power spectrum!
        # for consistent power spectrum
        self.norm_factor = 2 / self.sample_frequency / self.record_length  # / 0.875 # window factor
        self.norm_back_factor = 1 / self.norm_factor / self.record_length ** 2
        # for consistent amplitude
        self.norm_factor_amp = (2 / self.record_length) ** 2  # / 0.875 # window factor
        self.norm_back_factor_amp = 1 / self.norm_factor_amp / self.record_length ** 2

        self.Ce = self.C[self.tes_flag]
        # self.update_capacity()

        self.clear_buffer()

        assert self.tp_interval > 2 * self.record_length / self.sample_frequency, \
            'tp_interval must be longer than 2 times the record window'

    # ------------------------------
    # SETTER AND GETTER
    # ------------------------------

    def norm(self, value, range):  # from range to (-1,1),
        """
        docs missing
        """
        return 2 * (np.array(value) - range[0]) / (range[1] - range[0]) - 1

    def denorm(self, value, range):  # from (-1,1) to range,
        """
        docs missing
        """
        return range[0] + (np.array(value) + 1) / 2 * (range[1] - range[0])

    def set_control(self, dac, Ib, norm=False):
        """
        docs missing
        """
        assert len(dac) == self.nmbr_heater, ''
        assert len(Ib) == self.nmbr_tes, ''
        if norm:
            dac = self.denorm(dac, self.dac_range)
            Ib = self.denorm(Ib, self.Ib_range)
        self.dac = np.array(dac)
        self.Ib = np.array(Ib)

        # self.update_capacity()

    def get(self, name, norm=False, div_adc_by_bias=False):
        """
        docs missing
        """
        value = np.array(eval('self.' + name))
        if name in ['ph', 'rms', 'offset'] and div_adc_by_bias:
            for i in range(self.nmbr_tes):
                if self.Ib[i] > 0:
                    value[i] /= self.Ib[i]
        if norm:
            if name in ['ph', 'rms', 'offset']:
                value = self.norm(value, self.adc_range)
            if name == 'Ib':
                value = self.norm(value, self.Ib_range)
            elif name == 'tpa':
                value = self.norm(value, (0., np.max(self.tpa_queue)))
            else:
                value = self.norm(value, self.dac_range)
        return value

    def get_buffer(self, name):
        """
        docs missing
        """
        return np.array(eval('self.buffer_' + name))

    def get_record(self, no_noise=False):
        """
        Get the squid output record window.

        :return: The output of the squids, in shape (nmbr_tes, record_length).
        :rtype: numpy array
        """
        if no_noise:
            rec = np.array(self.squid_out)
        else:
            rec = np.array(self.squid_out_noise)
        return rec

    def get_temperatures(self):
        return np.array(self.T)

    # ------------------------------
    # PUBLIC METHODS
    # ------------------------------

    def calc_out(self):
        """
        docs missing
        """
        self.squid_out = self.eta * (self.Il - self.Il[0])  # remove offset
        if np.any(self.squid_out > self.adc_range[1]) or np.any(self.squid_out < self.adc_range[0]):
            if self.verb:
                print('ADC range exceeded!')
        self.squid_out[self.squid_out > self.adc_range[1]] = self.adc_range[1]
        self.squid_out[self.squid_out < self.adc_range[0]] = self.adc_range[0]
        self.squid_out_noise = np.copy(self.squid_out)

    def wait(self, seconds, update_T=True, two_steps=True):
        """
        docs missing
        attention, the recorded record window does not necessarily correspond to the integrated time
        integration is done twice, once with a mock heat capacity, once with the real one
        """
        self.timer += seconds
        self.update_capacitor(seconds)
        if update_T:
            self.pileup_t0 = None
            self.pileup_er = 0
            self.er = np.zeros(self.nmbr_components)
            self.tpa = np.zeros(self.nmbr_heater)

            mock_heat_cap = [lambda x: self.heat_capacity_facts[i](self.T[-1, c]) for i,c in zip(range(self.nmbr_tes), np.nonzero(self.tes_flag)[0])]
            if two_steps:
                iterator = zip([mock_heat_cap, self.heat_capacity_facts], [seconds, self.record_length / self.sample_frequency])
            else:
                iterator = zip([self.heat_capacity_facts], [seconds])
            for heat_cap, window in iterator:
                # C, Gb, Tb, G, P, Rs, Ib, Rt, L, tes_flag, timer, C_tdep
                args = (
                    self.C,
                    self.Gb,
                    self.Tb,
                    self.G,
                    self.P,
                    self.Rs,
                    self.Ib,
                    self.Rt,
                    self.L,
                    self.tes_flag,
                    self.timer,
                    heat_cap,
                )

                init_cond = np.concatenate((self.T[-1, :], self.It[-1].reshape(-1)))
                self.t = np.linspace(0, window, self.record_length)
                first_steps = self.t[:500:5]
                TIb = odeint(self.dTdItdt,
                             init_cond,
                             self.t, args=args,
                             tfirst=True,
                             tcrit=first_steps
                             )

                self.T = TIb[:, :self.nmbr_components]
                self.It = TIb[:, self.nmbr_components:]

            self.Il = self.Ib - self.It
            self.calc_out()  # does anything break if I dont do these two lines?
            self.calc_par()

    def trigger(self, er, tpa, store=True, time_passes=True):
        """
        docs missing
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
        if self.verb:
            print(f'T0 is {self.T[-1, :]} mK.')
        tstamp = time.time()

        mock_heat_cap = [lambda x: self.heat_capacity_facts[i](self.T[-1, c]) for i, c in
                         zip(range(self.nmbr_tes), np.nonzero(self.tes_flag)[0])]
        # C, Gb, Tb, G, P, Rs, Ib, Rt, L, tes_flag, timer, C_tdep
        args = (
            self.C,
            self.Gb,
            self.Tb,
            self.G,
            self.P,
            self.Rs,
            self.Ib,
            self.Rt,
            self.L,
            self.tes_flag,
            self.timer,
            mock_heat_cap,
        )

        time_step = 1 / self.sample_frequency
        first_steps = self.t[:500:5]
        tcrit = np.concatenate((np.arange(self.t0 - 3 * time_step,
                                                     self.t0 + np.maximum(np.max(self.lamb_tp), np.max(self.lamb)) * 10,
                                                     time_step), first_steps))
        tcrit = np.sort(tcrit)
        TIb = odeint(func=self.dTdItdt,
                     y0=np.concatenate((self.T[-1, :], self.It[-1].reshape(-1))),
                     t=self.t,
                     args=args,
                     tcrit=tcrit,
                     tfirst=True,
                     )
        # pdb.set_trace()

        self.T = TIb[:, :self.nmbr_components]
        self.It = TIb[:, self.nmbr_components:]
        self.Il = self.Ib - self.It
        self.calc_out()
        if self.verb:
            print(f'Calculated in {time.time() - tstamp} s.')
            tstamp = time.time()
        for c in np.arange(self.nmbr_tes):
            self.squid_out_noise[:, c] += self.get_noise_bl(tes_channel=c)
        if self.verb:
            print(f'Generated noise in {time.time() - tstamp} s.')
        self.calc_par()
        if store:
            self.append_buffer()
        if time_passes:
            self.timer += self.t[-1]
            self.update_capacitor(self.t[-1])

    def sweep_dac(self, start, end, heater_channel=0, norm=False):
        """
        docs missing
        """
        verb_save = self.verb
        self.verb = False
        if norm:
            start = self.denorm(start, self.dac_range)
            end = self.denorm(end, self.dac_range)
        for dac in tqdm(np.arange(start, end, np.sign(end - start) * self.dac_ramping_speed[0] * self.tp_interval)):
            self.dac[heater_channel] = np.array(dac)
            self.wait(seconds=self.tp_interval - self.record_length / self.sample_frequency)
            # self.update_capacity()
            self.trigger(er=np.zeros(self.nmbr_components), tpa=self.tpa_queue[self.tpa_idx]*np.ones(self.nmbr_heater),
                         store=True, time_passes=True)
            self.tpa_idx += 1
            if self.tpa_idx + 1 > len(self.tpa_queue):
                self.tpa_idx = 0
        self.verb = verb_save

    def sweep_Ib(self, start, end, tes_channel=0, norm=False):
        """
        docs missing
        """
        verb_save = self.verb
        self.verb = False
        if norm:
            start = self.denorm(start, self.Ib_range)
            end = self.denorm(end, self.Ib_range)
        for Ib in tqdm(np.arange(start, end, np.sign(end - start) * self.Ib_ramping_speed[0] * self.tp_interval)):
            self.Ib[tes_channel] = np.array(Ib)
            self.wait(seconds=self.tp_interval - self.record_length / self.sample_frequency)
            # self.update_capacity()
            self.trigger(er=np.zeros(self.nmbr_components), tpa=self.tpa_queue[self.tpa_idx]*np.ones(self.nmbr_heater),
                         store=True, time_passes=True)
            self.tpa_idx += 1
            if self.tpa_idx + 1 > len(self.tpa_queue):
                self.tpa_idx = 0
        self.verb = verb_save

    def send_testpulses(self, nmbr_tp=1):
        """
        docs missing
        """
        verb_save = self.verb
        self.verb = False
        for i in trange(nmbr_tp):
            self.wait(seconds=self.tp_interval - self.record_length / self.sample_frequency)
            # self.update_capacity()
            self.trigger(er=np.zeros(self.nmbr_components), tpa=self.tpa_queue[self.tpa_idx],
                         verb=False, store=True, time_passes=True)
            self.tpa_idx += 1
            if self.tpa_idx + 1 > len(self.tpa_queue):
                self.tpa_idx = 0
        self.verb = verb_save

    def clear_buffer(self):
        """
        docs missing
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
        self.buffer_pulses = []

    # ------------------------------
    # PLOTTERS
    # ------------------------------

    def plot_temperatures(self, xlim=None, show=True, save_path=None, dpi=None):
        """
        docs missing
        """

        fig, axes = plt.subplots(self.nmbr_components, 1, figsize=(10, 1.5 * self.nmbr_components), sharex=True,
                                 dpi=dpi)

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
            if xlim is not None:
                axes[i].set_xlim(xlim)
            ax_2 = axes[i].twinx()
            ax_2.plot(self.t, power_input[:, i], label='Heat input {} {} (keV / s)'.format(label, i), c='C3',
                      linewidth=2)
            ax_2.set_zorder(100)
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

    def plot_tes(self, xlim_temp=None, xlim_time=None, show=True, save_path=None, dpi=None):
        """
        docs missing
        """

        fig, axes = plt.subplots(self.nmbr_tes, 2, figsize=(10, 3 * self.nmbr_tes), dpi=dpi)
        axes = axes.flatten()

        for i in range(self.nmbr_tes):
            tes_channel = i
            t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]  # from tes idx to component idx

            # transition curve plot
            Tmin, Tmax = np.min(self.T[:, t_ch]), np.max(self.T[:, t_ch])
            Rmin, Rmax = self.Rt[tes_channel](Tmin), self.Rt[tes_channel](Tmax)
            temp = np.linspace(np.minimum(self.Tc[tes_channel] - 4 / self.k[tes_channel], np.minimum(Tmin, self.Tb(0))),
                               np.maximum(self.Tc[tes_channel] + 4 / self.k[tes_channel], Tmax), 1000)
            axes[2 * i + 0].plot(temp, 1000 * self.Rt[tes_channel](temp), label='Transition curve', c='#FF7979',
                                 linewidth=2)
            if xlim_temp is None:
                axes[2 * i + 0].axvline(x=self.Tb(0), color='grey', linestyle='dashed', label='Heat bath')
            else:
                if xlim_temp[0] <= self.Tb(0):
                    axes[2 * i + 0].axvline(x=self.Tb(0), color='grey', linestyle='dashed', label='Heat bath')
            axes[2 * i + 0].fill_between([Tmin, Tmax],
                                         [0, 0],
                                         [1000 * Rmin, 1000 * Rmax], color='#99CCFF', alpha=0.5, zorder=10)
            axes[2 * i + 0].fill_between([temp[0], Tmin, Tmax],
                                         [1000 * Rmin, 1000 * Rmin, 1000 * Rmax],
                                         [1000 * Rmax, 1000 * Rmax, 1000 * Rmax], color='#99CCFF', alpha=0.5, zorder=10)
            axes[2 * i + 0].plot([temp[0], Tmin], [1000 * Rmin, 1000 * Rmin], color='black',  # alpha=0.5,
                                 linewidth=2, label='OP', zorder=100)
            axes[2 * i + 0].plot([temp[0], Tmax], [1000 * Rmax, 1000 * Rmax], color='black',  # alpha=0.5,
                                 linewidth=2, zorder=100)
            axes[2 * i + 0].plot([Tmin, Tmin], [0, 1000 * Rmin], color='black',  # alpha=O.5,
                                 linewidth=2, zorder=100)
            axes[2 * i + 0].plot([Tmax, Tmax], [0, 1000 * Rmax], color='black',  # alpha=0.5,
                                 linewidth=2, zorder=100)
            axes[2 * i + 0].set_ylabel('Resistance (mOhm)')  # , c='#FF7979')
            axes[2 * i + 0].set_xlabel('Temperature (mK)')
            axes[2 * i + 0].legend(frameon=False).set_zorder(100)
            # axes[2 * i + 0].tick_params(axis='y', labelcolor='#FF7979')
            axes[2 * i + 0].set_title('TES curve {}'.format(i))
            if xlim_temp is not None:
                axes[2 * i + 0].set_xlim(xlim_temp)

            # recoil signature plot
            axes[2 * i + 1].plot(self.t, self.squid_out_noise[:, i], label='Squid output', zorder=5, c='black',
                                 linewidth=1,
                                 alpha=0.7)
            axes[2 * i + 1].plot(self.t, self.squid_out[:, i], label='Recoil signature', zorder=10, c='red',
                                 linewidth=2,
                                 alpha=1)
            axes[2 * i + 1].set_ylabel('Voltage (V)')  # , color='red'
            axes[2 * i + 1].set_xlabel('Time (s)')
            axes[2 * i + 1].legend(loc='upper right', frameon=False).set_zorder(100)
            axes[2 * i + 1].set_zorder(10)
            axes[2 * i + 1].set_title('Squid output {}'.format(i))
            if xlim_time is not None:
                axes[2 * i + 1].set_xlim(xlim_time)

        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            return fig, axes

    def plot_event(self, tes_channel=0, show=True):
        """
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

    def plot_nps(self, tes_channel=0, only_sum=False, save_path=None, show=True, dpi=None):
        """
        docs missing
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        Tt = self.T[0, t_ch]
        It = self.It[0, t_ch]

        fig, ax = plt.subplots(1, 1, figsize=(10, 3), dpi=dpi)

        w, nps = self.get_nps(Tt, It, tes_channel)
        h = self.get_lowpass(w)
        ax.loglog(w, 1e6 * np.sqrt(nps), label='combined', linewidth=2, color='black', zorder=10)

        if not only_sum:
            w = np.fft.rfftfreq(self.record_length, 1 / self.sample_frequency)
            h = h[w > 0]
            w = w[w > 0]
            nps = self.thermal_noise(w, Tt, It, tes_channel) * h
            ax.loglog(w, 1e6 * np.sqrt(nps), label='Thermal noise', linewidth=2)

            nps = self.thermometer_johnson(w, Tt, It, tes_channel) * h
            ax.loglog(w, 1e6 * np.sqrt(nps), label='Thermometer Johnson', linewidth=2)

            nps = self.shunt_johnson(w, Tt, It, tes_channel) * h
            ax.loglog(w, 1e6 * np.sqrt(nps), label='Shunt & excess Johnson', linewidth=2)

            nps = self.squid_noise(w, Tt, It, tes_channel) * h
            ax.loglog(w, 1e6 * np.sqrt(nps), label='Squid noise', linewidth=2)

            nps = self.one_f_noise(w, Tt, It, tes_channel) * h
            ax.loglog(w, 1e6 * np.sqrt(nps), label='1/f noise', linewidth=2)

            nps = self.emi_noise(w, Tt, It, tes_channel) * h
            ax.loglog(w, 1e6 * np.sqrt(nps), label='EM interference', linewidth=2)

        ax.set_title('Squid output {}'.format(tes_channel))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (pA / sqrt(Hz))')
        ax.legend(frameon=False, bbox_to_anchor=(1., 1.))

        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()

    def print_noise_parameters(self, channel=0):  # its the TES channel
        """
        docs missing
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][channel]
        Tt = self.T[0, t_ch]
        It = self.It[0, t_ch]

        print('Resistance TES / Resistance normal conducting: {}'.format(self.Rt[channel](Tt) / self.Rt0[channel]))
        print('Temperature mixing chamber: {} mK'.format(self.Tb(0)))
        print('Temperature TES: {} mK'.format(Tt))
        print('Resistance TES: {} mOhm'.format(1e3 * self.Rt[channel](Tt)))
        print('Tau eff: {} ms'.format(1e3 * self.tau_eff(Tt, It, channel)))
        print('TES Slope: {} mOhm/mK'.format(1e3 * self.dRtdT(Tt, channel)))
        print('C: {} fJ / K '.format(1e6 * self.C[t_ch] * self.heat_capacity_facts[t_ch](Tt)))
        print('Geff: {} pW / K'.format(1e3 * self.Gb[0] + 1e3 * self.G_etf(Tt, It, channel)))
        print('Tau in: {} ms'.format(1e3 * self.tau_in(Tt, channel)))
        print('Geb: {} pW / K '.format(1e3 * self.Gb[0]))
        print('G ETF: {} pW / K '.format(1e3 * self.G_etf(Tt, It, channel)))
        print('R shunt: {} mOhm'.format(1e3 * self.Rs[channel]))
        print('Temperature shunt: {} mK'.format(self.Tb(0)))
        print('i sq: {} pA/sqrt(Hz)'.format(self.i_sq[channel]))
        print('1 / f amplitude (pW): {} '.format(self.tes_fluct[channel]))
        print('1 / f power (flicker slope): {} '.format(self.flicker_slope[channel]))
        print('Tau el (s): {} '.format(self.tau_el(Tt, channel)))
        print('Tau I (s): {} '.format(self.tau_I(Tt, It, channel)))
        print('L_I (): {} '.format(self.L_I(Tt, It, channel)))

    def plot_buffer(self, tes_channel=0, tpa=None, save_path=None):
        """
        docs missing
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
        docs missing
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

    # ------------------------------
    # PHYSICS QUANTITIES THAT ARE METHODS
    # ------------------------------

    @staticmethod
    def Tb(t):
        """
        The time-dependent heat bath temperature.

        :param t: The timer time in seconds (e.g. self.timer).
        :type t: float
        :return: The heat bath temperature in ms.
        :rtype: float
        """
        T = 15.  # temp bath
        return T

    def update_capacitor(self, delta_t):
        """
        docs missing
        """
        self.U_sq_Rh = (self.U_sq_Rh - self.dac) * np.exp(- delta_t / self.tau_cap) + self.dac

    def P(self, t, T, It, no_pulses=False):
        """
        docs missing
        """

        P = np.zeros(T.shape)
        if t > self.t0 and not no_pulses:
            for i in range(self.nmbr_components):
                P += self.er[i] * self.eps[i] * np.exp(
                    -(t - self.t0) / self.lamb[i]) / self.lamb[i] * self.keV_to_pJ  # particle
        if self.pileup_t0 is not None and t > self.pileup_t0 and not no_pulses:
            P += self.pileup_er * self.eps[self.pileup_comp] * np.exp(
                -(t - self.pileup_t0) / self.lamb[self.pileup_comp]) / self.lamb[
                     self.pileup_comp] * self.keV_to_pJ  # pile up particle
        for i in range(self.nmbr_tes):
            c = np.nonzero(self.tes_flag)[0][i]
            P[c] += self.Rt[i](T[c]) * It[i] ** 2  # self heating
        for i in range(self.nmbr_heater):
            P += self.delta_h[i] * self.heater_current[i] ** 2 * (self.U_sq_Rh[i] / 10) * self.Rh[i]  # heating
        if t > self.t0 and not no_pulses:
            for i in range(self.nmbr_heater):
                P += (np.maximum(self.tpa[i], 0) / 10) * self.delta[i] * self.heater_current[i] ** 2 * \
                     self.pulser_scale[i] * np.exp(-(t - self.t0) / self.lamb_tp[i]) * self.Rh[i]  # test pulses
        return P

    # ------------------------------
    # PRIVATE METHODS
    # ------------------------------

    @staticmethod
    def dTdItdt(t, TIt, C, Gb, Tb, G, P, Rs, Ib, Rt, L, tes_flag, timer, C_tdep):
        """
        docs missing
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
        for i in range(Ib.shape[0]):
            c = np.nonzero(tes_flag)[0][i]
            dTdIt[i] /= C_tdep[c](T[c])  # update heat capacity

        # electrical
        dTdIt[nmbr_components:] = Rs * Ib  #
        for i in range(Ib.shape[0]):
            c = np.nonzero(tes_flag)[0][i]
            dTdIt[nmbr_components + i] -= It[i] * (Rt[i](T[c]) + Rs[i])  #
        dTdIt[nmbr_components:] /= L  # voltage to current

        return dTdIt

    def Rt_init(self, which_curve, k, Tc, Rt0):
        """
        docs missing
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
        docs missing
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

    # ------------------------------
    # NOISE FUNCTION GETTERS
    # ------------------------------

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

        noise = self.noise_function(nps, size=1)[0]  # in muA

        return self.eta[tes_channel] * noise  # in V

    def get_noise_bl_ou(self, tes_channel=0):
        """
        docs missing
        """

        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        Tt = self.T[0, t_ch]  # in mK
        It = self.It[0, t_ch]  # in muA

        # TODO integrate OU process

        noise = ...  # TODO  # in muA

        return self.eta[tes_channel] * noise  # in V

    def noise_function(self, nps, size):
        """
        A value trace in time space, that follows the given NPS.

        :param nps: The noise power spectrum in (muA)^2/Hz.
        :type nps: 1D numpy array
        :param size: The number of traces to simulate.
        :type size: int
        :return: The simulated noise trace in muA.
        :rtype: 1D numpy array
        """
        f = np.sqrt(nps * self.norm_back_factor)
        f = np.array(f, dtype='complex')  # create array for frequencies
        f = np.tile(f, (size, 1))
        phases = np.random.rand(size, nps.shape[0]) * 2 * np.pi  # create random phases
        phases = np.cos(phases) + 1j * np.sin(phases)
        f *= phases
        return np.fft.irfft(f, axis=-1, norm='forward')

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
            nps[w > 0] *= self.get_lowpass(w[w > 0])
        return w, nps

    # --------------------------------------------
    # PULSE SHAPE PARAMETERS AND TIME CONSTANTS
    # --------------------------------------------

    def get_thermal_time_constants(self, Tt=None, It=None, tes_channel=0):
        """
        docs missing
        this gives the exact eigenvalues of the absorber-thermometer matrix
        attention, tes should be component 0, absorber should be component 1
        """
        assert self.nmbr_components == 2, 'This formula is only valid for the 2 component model!'

        # tes should be component 0, absorber should be component 1
        Tt = self.T[0, 0] if Tt is None else Tt
        It = self.It[0, 0] if It is None else It
        tau_n = self.lamb[1]  # recoil in absorber
        G_ea = self.G[0, 1]
        G_eb = self.Gb[0] + self.G_etf(Tt, It)  # include ETF
        G_ab = self.Gb[1]
        C_e = self.C[0] * self.heat_capacity_facts[0](Tt)
        C_a = self.C[1]

        a = (G_ea + G_eb) / C_e + (G_ea + G_ab) / C_a
        b = (G_ea * G_eb + G_ea * G_ab + G_eb * G_ab) / (C_e * C_a)

        tau_eff = 2 / (a + np.sqrt(a ** 2 - 4 * b))
        tau_t = 2 / (a - np.sqrt(a ** 2 - 4 * b))

        return tau_n, tau_eff, tau_t

    def get_thermal_amplitudes(self, er, Tt=None, It=None, tes_channel=0):
        """
        docs missing
        """
        assert self.nmbr_components == 2, 'This formula is only valid for the 2 component model!'
        # tes should be component 0, absorber should be component 1
        Tt = self.T[0, 0] if Tt is None else Tt
        It = self.It[0, 0] if It is None else It
        tau_n, tau_eff, tau_t = self.get_thermal_time_constants(Tt, It, tes_channel)
        er = er * self.keV_to_pJ
        # tes should be component 0, absorber should be component 1
        eps = self.eps[1, 0]  # recoil in absorber, want share that thermalizes in thermometer
        tau_n = self.lamb[1]  # recoil in absorber
        G_ea = self.G[0, 1]
        G_ab = self.Gb[1]
        C_e = self.C[0] * self.heat_capacity_facts[0](Tt)
        C_a = self.C[1]

        alpha_in = 1 + G_ab / G_ea - 1 / tau_eff * C_a / G_ea
        alpha_t = 1 + G_ab / G_ea - 1 / tau_t * C_a / G_ea

        A_n = (er / tau_n)
        A_n /= (1 - alpha_t / alpha_in)
        A_n *= (alpha_t * (1 - eps) / C_a - eps / C_e)
        A_n /= (1 / tau_n - 1 / tau_eff)

        A_t = (-er / tau_n)
        A_t /= (1 - alpha_in / alpha_t)
        A_t *= (alpha_in * (1 - eps) / C_a - eps / C_e)
        A_t *= (1 / tau_n - 1 / tau_t) ** (-1)

        A_n_abs = A_n / alpha_in
        A_t_abs = A_t / alpha_t

        return A_n, A_t, A_n_abs, A_t_abs

    def get_thermal_pulseshape(self, er, Tt=None, It=None, tes_channel=0):
        """
        docs missing
        recoil energy in keV
        """
        assert self.nmbr_components == 2, 'This formula is only valid for the 2 component model!'
        t = np.arange(0, self.record_length / self.sample_frequency, 1 / self.sample_frequency, )
        t0 = self.t0

        A_n, A_t, A_n_abs, A_t_abs = self.get_thermal_amplitudes(er, Tt, It)
        tau_n, tau_eff, tau_t = self.get_thermal_time_constants(Tt, It, tes_channel)
        pulse_tes = A_n * (np.exp(-(t - t0) / tau_n) - np.exp(-(t - t0) / tau_eff)) + \
                    A_t * (np.exp(-(t - t0) / tau_t) - np.exp(-(t - t0) / tau_n))
        pulse_tes *= np.heaviside(t - self.t0, 1)
        pulse_abs = A_n_abs * (np.exp(-(t - t0) / tau_n) - np.exp(-(t - t0) / tau_eff)) + \
                    A_t_abs * (np.exp(-(t - t0) / tau_t) - np.exp(-(t - t0) / tau_n))
        pulse_abs *= np.heaviside(t - self.t0, 1)
        return pulse_tes, pulse_abs

    def tau_in(self, Tt, tes_channel=0):
        """
        docs missing
        attention, this formula ignores the absorber! (Gea=0)
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        return self.C[t_ch] * self.heat_capacity_facts[t_ch](Tt) / self.Gb[t_ch]

    def tau_eff(self, Tt, It, tes_channel=0):
        """
        docs missing
        attention, this formula ignores the absorber! (Gea=0)
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        return self.tau_in(Tt, tes_channel) / (1 + self.G_etf(Tt, It) / self.Gb[t_ch])

    def tau_t(self, abs_channel=1):
        """
        docs missing
        attention, this formula ignores the thermometer! (Gea=0)
        """
        return self.C[abs_channel] / self.Gb[abs_channel]

    def G_etf(self, Tt, It, tes_channel=0):
        """
        docs missing
        """
        return It ** 2 * self.dRtdT(Tt, tes_channel) * (self.Rt[tes_channel](Tt) - self.Rs[tes_channel]) / (
                self.Rt[tes_channel](Tt) + self.Rs[tes_channel])

    def tau_el(self, Tt, tes_channel=0):
        """
        docs missing
        """
        return self.L[tes_channel] / (self.Rs[tes_channel] + self.Rt[tes_channel](Tt))

    def L_I(self, Tt, It, tes_channel=0):
        """
        docs missing
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        return It ** 2 / self.Gb[t_ch] * self.dRtdT(Tt, tes_channel)

    def tau_I(self, Tt, It, tes_channel=0):
        """
        docs missing
        """
        return self.tau_in(Tt, tes_channel) / (1 - self.L_I(Tt, It, tes_channel))

    # ------------------------------
    # FITTER
    # ------------------------------

    def solve_thermal_ps(self, fitpars, er, rranges, Tt=None, It=None, print_iv=1000):
        """
        docs missing
        attention, this formula only valid for the 2 component model!

        well this function does not work so nicely after all, its probably too much to fit all things at once
        """
        fA_n, fA_t, ftau_eff, ftau_t = fitpars

        Tt = self.T[0, 0] if Tt is None else Tt
        It = self.It[0, 0] if It is None else It

        counter = [0]
        lowest_loss = [1e10]
        start_time = time.time()

        def func(p, counter, lowest_loss):
            G_eb, G_ab, G_ea, eps = p
            self.Gb[0] = np.abs(G_eb)  # TES channel
            self.Gb[1] = np.abs(G_ab)  # Absorber channel
            self.G[0, 1] = np.abs(G_ea)  # electron phonon coupling channel
            self.G[1, 0] = np.abs(G_ea)
            self.eps[1, 0] = np.abs(eps)  # collection efficiency
            self.eps[1, 1] = 1 - np.abs(eps)
            A_n, A_t, A_n_abs, A_t_abs = self.get_thermal_amplitudes(er, Tt, It)
            tau_n, tau_eff, tau_t = self.get_thermal_time_constants(Tt, It, 0)
            eqs = np.array([A_n - fA_n, A_t - fA_t, tau_eff - ftau_eff, tau_t - ftau_t])
            loss = np.sum(eqs ** 2)
            if np.isnan(loss) or np.isinf(loss):
                loss = 1e10
            if loss < lowest_loss[0]:
                lowest_loss[0] = loss
            if (counter[0] + 1) % print_iv == 0:
                print('done: {}, total: {}, {} %, time passed: {}'.format((counter[0] + 1), nmbr_steps,
                                                                          100 * (counter[0] + 1) / nmbr_steps,
                                                                          time.time() - start_time))
                print('lowest loss: {}, G_eb, G_ab, G_ea, eps = {}, {}, {}, {}'.format(lowest_loss[0], G_eb, G_ab, G_ea,
                                                                                       eps))
            counter[0] += 1
            return loss

        nmbr_steps = np.product([(r.stop - r.start) / r.step for r in rranges])
        resbrute = brute(func, rranges, args=(counter, lowest_loss,), full_output=True, finish=fmin)
        return resbrute

    def solve_time_constants(self, fitpars, rranges, Tt=None, It=None, print_iv=1000):
        """
        docs missing
        attention, this formula only valid for the 2 component model!
        """
        ftau_eff, ftau_t = fitpars

        Tt = self.T[0, 0] if Tt is None else Tt
        It = self.It[0, 0] if It is None else It

        counter = [0]
        lowest_loss = [1e10]
        best_couplings = [0, 0, 0]
        start_time = time.time()

        def func(p, counter, lowest_loss, best_couplings):
            G_eb, G_ab, G_ea = p
            self.Gb[0] = np.abs(G_eb)
            self.Gb[1] = np.abs(G_ab)
            self.G[0, 1] = np.abs(G_ea)
            self.G[1, 0] = np.abs(G_ea)
            tau_n, tau_eff, tau_t = self.get_thermal_time_constants(Tt, It, 0)
            eqs = np.array([tau_eff - ftau_eff, tau_t - ftau_t])
            loss = np.sum(eqs ** 2)
            if np.isnan(loss) or np.isinf(loss):
                loss = 1e10
            if loss < lowest_loss[0]:
                lowest_loss[0] = loss
                best_couplings[0], best_couplings[1], best_couplings[2] = G_eb, G_ab, G_ea
            if (counter[0] + 1) % print_iv == 0:
                print('done: {}, total: {}, {} %, time passed: {}'.format(counter[0]+1, nmbr_steps,
                                                                          100 * (counter[0]+1) / nmbr_steps,
                                                                          time.time() - start_time))
                print('lowest loss: {}, G_eb, G_ab, G_ea = {}, {}, {}'.format(lowest_loss[0], *best_couplings))
            counter[0] += 1
            return loss

        nmbr_steps = np.product([(r.stop - r.start) / r.step for r in rranges])
        resbrute = brute(func, rranges, args=(counter, lowest_loss, best_couplings), full_output=True, finish=fmin)
        return resbrute

    def solve_dac(self, R_op, limits):
        """
        docs missing
        attention, this formula only valid for the 2 component model!
        """

        def froot_dac(dac):
            self.set_control(dac=np.array([dac, ]), Ib=self.Ib, norm=False)
            self.wait(5)
            rt = self.Rt[0](self.T[-1, 0])
            return rt - R_op

        verbmem = self.verb
        self.verb = False
        try:
            res = brentq(froot_dac, limits[0], limits[1], rtol=1e-4)
        except ValueError:
            print('No different signs: ', froot_dac(limits[0]), froot_dac(limits[1]))
            res = np.nan
        self.verb = verbmem

        return res

    def solve_Rh(self, R_op, limits):
        """
        docs missing
        attention, this formula only valid for the 2 component model!
        """

        def froot_Rh(Rh):
            self.Rh = np.array([Rh, ])
            self.wait(5)
            rt = self.Rt[0](self.T[-1, 0])
            return rt - R_op

        verbmem = self.verb
        self.verb = False
        try:
            res = brentq(froot_Rh, limits[0], limits[1], rtol=1e-4)
        except ValueError:
            print('No different signs: ', froot_Rh(limits[0]), froot_Rh(limits[1]))
            res = np.nan
        self.verb = verbmem

        return res

    def solve_collection_efficiency(self, er, ph_cal, limits):
        """
        docs missing
        attention, this formula only valid for the 2 component model!
        """

        def froot_eps(eps):
            self.eps = np.array([[0.99, 0.01], [eps, 1 - eps], ])
            self.wait(5)
            self.trigger(er=np.array([0., er]), tpa=np.array([0.0]))
            ev = self.get_record(no_noise=True)
            ph = np.max(ev) - np.mean(ev[:2000])
            return ph - ph_cal

        verbmem = self.verb
        self.verb = False
        try:
            res = brentq(froot_eps, a=limits[0], b=limits[1], rtol=1e-4)
        except ValueError:
            print('No different signs: ', froot_eps(limits[0]), froot_eps(limits[1]))
            res = np.nan
        self.verb = verbmem

        return res

    def solve_delta(self, tpa, ph_tpa, limits):
        """
        docs missing
        attention, this formula only valid for the 2 component model!
        """

        def froot_delta(delta):
            self.delta = np.array([[delta, 1 - delta], ])
            self.wait(5)
            self.trigger(er=np.array([0., 0.]), tpa=np.array([tpa]))
            ev = self.get_record(no_noise=True)
            ph = np.max(ev) - np.mean(ev[:2000])
            return ph - ph_tpa

        verbmem = self.verb
        self.verb = False
        try:
            res = brentq(froot_delta, a=limits[0], b=limits[1], rtol=1e-4)
        except ValueError:
            print('No different signs: ', froot_delta(limits[0]), froot_delta(limits[1]))
            res = np.nan
        self.verb = verbmem

        return res

    def solve_pulser_scale(self, tpa, ph_tpa, limits):
        """
        docs missing
        attention, this formula only valid for the 2 component model!
        """

        def froot_psc(psc):
            self.pulser_scale = np.array([psc])
            self.wait(5)
            self.trigger(er=np.array([0., 0.]), tpa=np.array([tpa]))
            ev = self.get_record(no_noise=True)
            ph = np.max(ev) - np.mean(ev[:2000])
            return ph - ph_tpa

        verbmem = self.verb
        self.verb = False
        try:
            res = brentq(froot_psc, a=limits[0], b=limits[1], rtol=1e-4)
        except ValueError:
            print('No different signs: ', froot_psc(limits[0]), froot_psc(limits[1]))
            res = np.nan
        self.verb = verbmem

        return res

    def solve_tes_fluctuations(self, nps, Tt=None, It=None, tes_channel=0, idx=1):

        Tt = self.T[0, 0] if Tt is None else Tt
        It = self.It[0, 0] if It is None else It

        w, nps_calc = self.get_nps(Tt, It, tes_channel=tes_channel)
        self.tes_fluct[tes_channel] = np.sqrt(nps[idx] - nps_calc[idx])

        return self.tes_fluct[tes_channel]

    def solve_johnson_excess(self, nps, Tt=None, It=None, tes_channel=0, idx=1000):

        Tt = self.T[0, 0] if Tt is None else Tt
        It = self.It[0, 0] if It is None else It

        w, nps_calc = self.get_nps(Tt, It, tes_channel=tes_channel)
        self.excess_johnson[tes_channel] = np.sqrt(nps[idx] / nps_calc[idx])

        return self.excess_johnson[tes_channel]

    def solve_emi(self, nps, Tt=None, It=None, tes_channel=0):

        Tt = self.T[0, 0] if Tt is None else Tt
        It = self.It[0, 0] if It is None else It

        w, nps_calc = self.get_nps(Tt, It, tes_channel=tes_channel)

        idx0 = (np.abs(w - 50)).argmin()
        idx1 = (np.abs(w - 100)).argmin()
        idx2 = (np.abs(w - 150)).argmin()

        self.emi[tes_channel][0] = np.sqrt(nps[idx0] - nps_calc[idx0])
        self.emi[tes_channel][1] = np.sqrt(nps[idx1] - nps_calc[idx1])
        self.emi[tes_channel][2] = np.sqrt(nps[idx2] - nps_calc[idx2])

        return self.emi[tes_channel]

    # ------------------------------
    # NOISE FUNCTIONS
    # ------------------------------

    def get_lowpass(self, w):
        """
        docs missing
        """
        b, a = butter(N=1, Wn=self.lowpass, btype='lowpass', analog=True)
        _, h = freqs(b, a, worN=w)
        return np.abs(h) ** 2

    def dRtdT(self, T, tes_channel=0):
        """
        docs missing
        """
        grid = np.linspace(self.Tc[tes_channel] - 5 / self.k[tes_channel],
                           self.Tc[tes_channel] + 5 / self.k[tes_channel], 1000)
        h = (grid[1] - grid[0])
        return np.interp(T, grid[:-1] + h / 2, 1 / h * np.diff(self.Rt[tes_channel](grid)))

    def power_to_current_int(self, w, Tt, It, tes_channel=0):
        """
        docs missing
        for phonon noise
        """
        s21_int = -1 / (It)
        denom = self.L[tes_channel] / (self.tau_el(Tt, tes_channel) * self.L_I(Tt, It, tes_channel))
        denom += (self.Rt[tes_channel](Tt) + self.Rs[tes_channel])  # here the sign other than in Irwin
        denom += 2 * np.pi * 1j * w * self.L[tes_channel] * self.tau_in(Tt, tes_channel) / self.L_I(Tt, It,
                                                                                                    tes_channel) * (
                         1 / self.tau_in(Tt, tes_channel) + 1 / self.tau_el(Tt, tes_channel))
        denom -= (4 * np.pi ** 2 * w ** 2 * self.tau_in(Tt, tes_channel) * self.L[tes_channel]) / self.L_I(Tt, It,
                                                                                                           tes_channel)
        s21_int /= denom
        return s21_int

    def volt_to_current_int(self, w, Tt, It, tes_channel=0):
        """
        docs missing
        for Johnson shunt and 1/f
        """
        s21_int = -1 / (It)
        denom = self.L[tes_channel] / self.tau_el(Tt, tes_channel)
        denom += (self.Rt[tes_channel](Tt) + self.Rs[tes_channel]) * self.L_I(Tt, It,
                                                                              tes_channel)  # here the sign other than in Irwin
        denom += 2 * np.pi * 1j * w * self.L[tes_channel] * self.tau_in(Tt, tes_channel) * (
                1 / self.tau_in(Tt, tes_channel) + 1 / self.tau_el(Tt, tes_channel))
        denom -= (4 * np.pi ** 2 * w ** 2 * self.tau_in(Tt, tes_channel) * self.L[tes_channel])
        s21_int /= denom

        s22_int = -s21_int * It * (1 + 2 * np.pi * 1j * w * self.tau_in(Tt, tes_channel))
        return s22_int

    def power_to_current_ext(self, w, Tt, It, tes_channel=0):
        """
        docs missing
        for Johnson TES
        """
        s21_ext = -1 / (It)
        denom = self.L[tes_channel] * self.tau_in(Tt, tes_channel) / (
                self.tau_el(Tt, tes_channel) * self.tau_I(Tt, It, tes_channel) * self.L_I(Tt, It, tes_channel))
        denom += 2 * self.Rt[tes_channel](Tt)
        denom += 2 * np.pi * 1j * w * self.L[tes_channel] * self.tau_in(Tt, tes_channel) / self.L_I(Tt, It,
                                                                                                    tes_channel) * (
                         1 / self.tau_I(Tt, It, tes_channel) + 1 / self.tau_el(Tt, tes_channel))
        denom -= (4 * np.pi ** 2 * w ** 2 * self.tau_in(Tt, tes_channel)) / self.L_I(Tt, It, tes_channel) * self.L[
            tes_channel]
        s21_ext /= denom
        return s21_ext

    def volt_to_current_ext(self, w, Tt, It, tes_channel=0):
        """
        docs missing
        for Johnson TES and flicker
        """
        s21_ext = self.power_to_current_ext(w, Tt, It, tes_channel)
        s22_ext = s21_ext * It * (self.L_I(Tt, It, tes_channel) - 1) / self.L_I(Tt, It, tes_channel) * (
                1 + 2 * np.pi * 1j * w * self.tau_I(Tt, It, tes_channel))
        return s22_ext

    def thermal_prefactor(self, Tt, tes_channel=0):
        """
        docs missing
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        thermal_pref = 4 * self.kB * Tt ** 2 * self.Gb[t_ch] * 2 / 5 * (
                1 - (self.Tb(0) / Tt) ** 5) / (1 - (self.Tb(0) / Tt) ** 2) * 1e3
        return thermal_pref  # pW

    def johnson_prefactor(self, T, R):
        """
        docs missing
        """
        jf_pref = 4 * self.kB * T * R * 1e3
        return jf_pref  # in pV

    def thermal_noise(self, w, Tt, It, tes_channel=0):
        """
        docs missing
        return in (muA)^2/Hz
        """
        if self.L_I(Tt, It, tes_channel) > 0:
            noise = np.abs(self.power_to_current_int(w, Tt, It, tes_channel)) ** 2
            noise *= self.thermal_prefactor(Tt, tes_channel)
            noise *= self.excess_phonon[tes_channel] ** 2
        else:
            noise = 0 * w / w
        return noise

    def thermometer_johnson(self, w, Tt, It, tes_channel=0):
        """
        docs missing
        return in (muA)^2/Hz
        """
        if self.L_I(Tt, It, tes_channel) > 0:
            noise = np.abs(self.volt_to_current_ext(w, Tt, It, tes_channel)) ** 2
            noise *= self.johnson_prefactor(Tt, self.Rt[tes_channel](Tt))
        else:
            noise = 0 * w / w
        return noise

    def shunt_johnson(self, w, Tt, It, tes_channel=0):
        """
        docs missing
        return in (muA)^2/Hz
        """
        noise = np.abs(self.volt_to_current_int(w, Tt, It, tes_channel)) ** 2
        noise *= self.johnson_prefactor(self.Tb(0), self.Rs[tes_channel])
        noise *= self.excess_johnson[tes_channel] ** 2
        return noise

    def squid_noise(self, w, Tt, It, tes_channel=0):
        """
        docs missing
        return in (muA)^2/Hz
        """
        noise = (self.i_sq[tes_channel]) ** 2 * w / w * 1e-12
        return noise

    def one_f_noise(self, w, Tt, It, tes_channel=0):
        """
        docs missing
        return in (muA)^2/Hz
        """
        if self.L_I(Tt, It, tes_channel) > 0:
            noise = np.abs(self.volt_to_current_ext(w, Tt, It, tes_channel)) ** 2
            noise *= 1 / np.maximum(w, 1) ** (self.flicker_slope[tes_channel])
            noise *= It ** 2 * self.Rt[tes_channel](Tt) ** 2
            noise *= self.tes_fluct[tes_channel] ** 2
        else:
            noise = 0 * w / w
        return noise

    def emi_noise(self, w, Tt, It, tes_channel=0):
        """
        docs missing
        return in (muA)^2/Hz
        """

        self.t = np.arange(0, self.record_length / self.sample_frequency, 1 / self.sample_frequency)
        power_ts = np.sin(2 * np.pi * self.t * 50)
        power_component = np.abs(np.fft.rfft(power_ts)) ** 2 * self.norm_factor_amp

        power_ts_two = np.sin(2 * np.pi * self.t * 150)
        power_component_two = np.abs(np.fft.rfft(power_ts_two)) ** 2 * self.norm_factor_amp

        power_ts_three = np.sin(2 * np.pi * self.t * 250)
        power_component_three = np.abs(np.fft.rfft(power_ts_three)) ** 2 * self.norm_factor_amp

        noise = np.abs(self.volt_to_current_int(w, Tt, It, tes_channel)) ** 2

        idx0 = np.argmax(power_component)
        idx1 = np.argmax(power_component_two)
        idx2 = np.argmax(power_component_three)

        p0_norm = 1 / power_component[idx0] * power_component / noise[idx0]
        p1_norm = 1 / power_component_two[idx1] * power_component_two / noise[idx1]
        p2_norm = 1 / power_component_three[idx2] * power_component_three / noise[idx2]

        noise *= (self.emi[tes_channel][0] ** 2 * p0_norm[-w.shape[0]:] +
                  self.emi[tes_channel][1] ** 2 * p1_norm[-w.shape[0]:] +
                  self.emi[tes_channel][2] ** 2 * p2_norm[-w.shape[0]:])

        return noise

    # ------------------------------
    # OLD NOISE FUNCTIONS
    # ------------------------------

    def thermal_noise_tot(self, w, Tt, It, tes_channel=0):
        """
        return in (muA)^2/Hz
        attention, deprecated, just for backward reference!
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        P_th_squared = 4 * self.kB * Tt ** 2 * self.Gb[t_ch]
        if Tt > self.Tb(0):
            P_th_squared *= 2 / 5 * (1 - (self.Tb(0) / Tt) ** 5) / (1 - (self.Tb(0) / Tt) ** 2)
        S_squared = 1 / (1 + w ** 2 * self.tau_eff(Tt, It, tes_channel) ** 2)
        S_squared /= (self.Gb[t_ch] + self.G_etf(Tt, It, tes_channel)) ** 2
        S_squared *= (It / (self.Rt[tes_channel](Tt) + self.Rs[tes_channel])) ** 2
        S_squared *= self.dRtdT(Tt, tes_channel) ** 2
        return S_squared * P_th_squared * 1e3

    def thermometer_johnson_tot(self, w, Tt, It, tes_channel=0):
        """
        return in (muA)^2/Hz
        attention, deprecated, just for backward reference!
        """
        I_jt = (1 + w ** 2 * self.tau_in(tes_channel) ** 2)
        I_jt /= (1 + w ** 2 * self.tau_eff(Tt, It, tes_channel) ** 2)
        I_jt *= 4 * self.kB * Tt * self.Rt[tes_channel](Tt) / (self.Rt[tes_channel](Tt) + self.Rs[tes_channel]) ** 2
        I_jt *= (self.tau_eff(Tt, It, tes_channel) / self.tau_in(tes_channel)) ** 2
        return I_jt * 1e3

    def shunt_johnson_tot(self, w, Tt, It, tes_channel=0):
        """
        return in (muA)^2/Hz
        attention, deprecated, just for backward reference!
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        I_js = (1 - It ** 2 / self.Gb[t_ch] * self.dRtdT(Tt, tes_channel)) ** 2 + w ** 2 * self.tau_in(tes_channel) ** 2
        I_js /= (1 + w ** 2 * self.tau_eff(Tt, It, tes_channel) ** 2)
        I_js *= 4 * self.kB * self.Tb(self.timer) * self.Rs[tes_channel] / (
                self.Rt[tes_channel](Tt) + self.Rs[tes_channel]) ** 2
        I_js *= (self.tau_eff(Tt, It, tes_channel) / self.tau_in(tes_channel)) ** 2
        return I_js * 1e3

    def squid_noise_tot(self, w, Tt, It, tes_channel=0):
        """
        return in (muA)^2/Hz
        attention, deprecated, just for backward reference!
        """
        I_sq = w / w
        I_sq *= (self.i_sq[tes_channel] * 1e-12) ** 2
        return I_sq * 1e-12

    def one_f_noise_tot(self, w, Tt, It, tes_channel=0):
        """
        return in (muA)^2/Hz
        attention, deprecated, just for backward reference!
        """
        I_one_f = (1 + w ** 2 * self.tau_in(tes_channel) ** 2)
        I_one_f /= (1 + w ** 2 * self.tau_eff(Tt, It, tes_channel) ** 2)
        I_one_f *= (self.tau_eff(Tt, It, tes_channel) / self.tau_in(tes_channel)) ** 2
        I_one_f *= It ** 2 * self.Rt[tes_channel](Tt) ** 2 * (self.tes_fluct[tes_channel]) ** 2 / w ** \
                   self.flicker_slope[tes_channel] / (
                           self.Rt[tes_channel](Tt) + self.Rs[tes_channel]) ** 2
        return I_one_f * 1e12

    def emi_noise_tot(self, w, Tt, It, tes_channel=0):
        """
        return in (muA)^2/Hz
        attention, deprecated, just for backward reference!
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        I_em = (1 - It ** 2 / self.Gb[t_ch] * self.dRtdT(Tt, tes_channel)) ** 2 + w ** 2 * self.tau_in(tes_channel) ** 2
        I_em /= (1 + w ** 2 * self.tau_eff(Tt, It, tes_channel) ** 2)
        I_em /= (self.Rt[tes_channel](Tt) + self.Rs[tes_channel]) ** 2
        I_em *= (self.tau_eff(Tt, It, tes_channel) / self.tau_in(tes_channel)) ** 2
        I_em *= (self.emi[tes_channel][0]) ** 2 * self.power_freq[-w.shape[0]:]
        return I_em * 1e3

    # ------------------------------
    # UTILS
    # ------------------------------

    def calc_par(self):
        """
        docs missing
        """
        self.offset = np.mean(self.squid_out_noise[self.t < self.t0], axis=0)
        self.ph = np.max(self.squid_out_noise[self.t >= self.t0] - self.offset, axis=0)
        self.rms = np.std(self.squid_out_noise[self.t < self.t0] - self.offset, axis=0)

    def append_buffer(self):
        """
        docs missing
        Attention, the dac and tes channels are wrongly aligned! (problem for more components)
        Not sure how to fix this honestly
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
            self.nmbr_tes))  # buffer not okay for different number of heaters and tes!
        if self.store_raw:
            pulse = self.squid_out_noise.reshape((256, -1, self.nmbr_tes))  # down sample for storage
            pulse = np.mean(pulse, axis=1)
            self.buffer_pulses.extend(pulse.T)

        for ls in [self.buffer_offset, self.buffer_ph, self.buffer_rms,
                   self.buffer_dac, self.buffer_Ib, self.buffer_tpa,
                   self.buffer_timer, self.buffer_channel, self.buffer_tes_resistance,
                   self.buffer_pulses]:
            while len(ls) > self.max_buffer_len:
                del ls[0]

    def reset_state(self):

        self.t = np.arange(0, self.record_length / self.sample_frequency, 1 / self.sample_frequency)  # s
        self.tpa_idx = 0
        self.timer = 0
        self.T = self.Tb(0) * np.ones((self.record_length, self.nmbr_components))
        self.Il, self.It = self.currents(self.T[:, self.tes_flag])
        self.calc_out()
        self.dac[:] = 0.
        self.Ib[:] = 0.
