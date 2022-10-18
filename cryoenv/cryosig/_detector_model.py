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


class DetectorModule:
    """
    TODO
    """

    def __init__(self,
                 record_length=16384,
                 sample_frequency=25000,
                 C=np.array([5e-5, 5e-4]),  # pJ / mK
                 Gb=np.array([5e-3, 5e-3]),  # pW / mK
                 G=np.array([[0., 1e-3], [1e-3, 0.], ]),  # heat cond between components, pW / mK
                 lamb=0.003,  # thermalization time (s)
                 eps=np.array([0.1, (1 - 0.1), ]),  # share thermalization in components
                 delta=np.array([0.02, (1 - 0.02), ]),  # share thermalization in components
                 Rs=np.array([0.035]),  # Ohm
                 Rh=np.array([10]),  # Ohm
                 Rt0=0.2,  # Ohm
                 L=np.array([3.5e-7]),  # H
                 k=2.,  # 1/mK
                 Tc=15.,  # mK
                 Ib=1.,  # muA
                 dac=np.array([0.]),  # V
                 pulser_scale=1.,  # scale factor
                 heater_attenuator=1.,
                 tes_flag=np.array([True, False], dtype=bool),  # which component is a tes
                 heater_flag=np.array([False, True], dtype=bool),  # which component has a heater
                 t0=.16,  # onset of the trigger, s
                 pileup_prob=0.02,  # percent / record window
                 tpa_queue=np.array([0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ]),  # V
                 tp_interval=5,  # s
                 max_buffer_len=10000,
                 dac_ramping_speed=np.array([2e-3]),  # V / s
                 Ib_ramping_speed=np.array([5e-3]),  # muA / s
                 xi=1.,  # squid conversion current to voltage, V / muA
                 i_sq=np.array([2 * 1e-12]),  # squid noise, A / sqrt(Hz)
                 tes_fluct=np.array([2e-3]),  # percent
                 lowpass=1e4,  # Hz
                 Tb=None,  # function that takes one positional argument t, returns Tb
                 Rt=None,  # function that takes one positional argument T, returns Rt
                 dac_range=(0., 5.),  # V
                 Ib_range=(1e-1, 1e1),  # muA
                 adc_range=(-10., 10.),  # V
                 store_raw=True,
                 ):

        # TODO fix for multiple channels
        # TODO plot instable regions in DAC/Ib
        # TODO absorber & heater noise
        # TODO temp dependence couplings and capacities

        # TODO! Gym wrapper
        # TODO! test case with SAC agent

        self.C = C
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
        self.pulser_scale = pulser_scale
        self.heater_attenuator = heater_attenuator
        self.tes_flag = tes_flag
        self.heater_flag = heater_flag
        self.record_length = record_length
        self.sample_frequency = sample_frequency
        self.t0 = t0
        self.pileup_prob = pileup_prob
        self.tpa_queue = tpa_queue
        self.tp_interval = tp_interval
        self.max_buffer_len = max_buffer_len
        self.dac_ramping_speed = dac_ramping_speed
        self.Ib_ramping_speed = Ib_ramping_speed
        self.xi = xi
        self.i_sq = i_sq
        self.tes_fluct = tes_fluct
        self.lowpass = lowpass
        if Tb is not None:
            self.Tb = Tb
        if Rt is not None:
            self.Rt = Rt
        else:
            self.Rt = self.Rt_init()
        self.dac_range = dac_range
        self.Ib_range = Ib_range
        self.adc_range = adc_range

        self.nmbr_components = len(self.C)
        self.nmbr_tes = len(self.Rs)
        self.nmbr_heater = len(self.Rh)
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

        self.Ce = self.C[self.tes_flag]
        self.C[self.tes_flag] = self.Ce * (2.43 - 1.43 * self.Rt(self.T[0, self.tes_flag]) / self.Rt0)

        self.clear_buffer()

    # setter and getter

    def norm(self, value, range):  # from range to (-1,1)
        return 2 * (np.array(value) - range[0]) / (range[1] - range[0]) - 1

    def denorm(self, value, range):  # from (-1,1) to range
        return range[0] + (np.array(value) + 1) / 2 * (range[1] - range[0])

    def set_control(self, dac, Ib, norm=True):
        """
        TODO
        """
        if norm:
            dac = self.denorm(dac, self.dac_range)
            Ib = self.denorm(Ib, self.Ib_range)
        self.dac = np.array(dac)
        self.Ib = np.array(Ib)

        self.C[self.tes_flag] = self.Ce * (2.43 - 1.43 * self.Rt(self.T[0, self.tes_flag]) / self.Rt0)

    def get(self, name, norm=True):
        value = np.array(eval('self.' + name))
        if norm:
            if name in ['ph', 'rms', 'offset']:
                value = self.norm(value, self.adc_range)
            else:
                value = self.norm(value, eval('self.' + name + '_range'))
        return value

    def get_buffer(self, name):
        """
        TODO
        """
        return np.array(eval('self.buffer_' + name))

    def get_record(self):
        """
        TODO
        """
        return np.array(self.squid_out_noise)

    # public

    def calc_out(self):
        self.squid_out = self.xi * (self.Il - self.Il[0])
        self.squid_out_noise = np.copy(self.squid_out)
        self.squid_out[self.squid_out > self.adc_range[1]] = self.adc_range[1]
        self.squid_out[self.squid_out < self.adc_range[0]] = self.adc_range[0]
        self.squid_out_noise[self.squid_out_noise > self.adc_range[1]] = self.adc_range[1]
        self.squid_out_noise[self.squid_out_noise < self.adc_range[0]] = self.adc_range[0]

    def wait(self, seconds, update_T=True):
        """
        TODO
        """
        self.timer += seconds
        if update_T:
            self.pileup_t0 = None
            self.pileup_er = 0
            self.er = 0
            self.tpa = 0
            self.t = np.linspace(0, seconds, self.record_length)

            TIb = odeint(self.dTdItdt,
                         np.concatenate((self.T[-1, :], self.It[-1].reshape(-1))),
                         self.t, args=(
                    self.C, self.Gb, self.Tb, self.G, self.P, self.Rs, self.Ib, self.Rt, self.L, self.tes_flag),
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
                     args=(self.C, self.Gb, self.Tb, self.G, self.P, self.Rs, self.Ib, self.Rt, self.L, self.tes_flag),
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
            self.squid_out_noise[:, c] += self.get_noise_bl(channel=c)
        if verb:
            print(f'Generated noise in {time.time() - tstamp} s.')
        self.calc_par()
        if store:
            self.append_buffer()
        if time_passes:
            self.timer += self.t[-1]

    def sweep_dac(self, start, end, norm=True):
        """
        TODO
        """
        if norm:
            start = self.denorm(start, self.dac_range)
            end = self.denorm(end, self.dac_range)
        dac_values = []
        for s, e in zip(start, end):
            dac_values.append(np.arange(s, e, np.sign(e - s) * self.dac_ramping_speed[0] * self.tp_interval))
        for dac in tqdm(zip(*dac_values), total=dac_values[0].shape[0]):
            self.dac = np.array(dac)
            self.wait(seconds=self.tp_interval - self.t[-1])
            self.C[self.tes_flag] = self.Ce * (2.43 - 1.43 * self.Rt(self.T[0, self.tes_flag]) / self.Rt0)
            self.trigger(er=0, tpa=self.tpa_queue[self.tpa_idx],
                         verb=False, store=True, time_passes=True)
            self.tpa_idx += 1
            if self.tpa_idx + 1 > len(self.tpa_queue):
                self.tpa_idx = 0

    def sweep_Ib(self, start, end, norm):
        """
        TODO
        """
        if norm:
            start = self.denorm(start, self.Ib_range)
            end = self.denorm(end, self.Ib_range)
        Ib_values = []
        for s, e in zip(start, end):
            Ib_values.append(np.arange(s, e, np.sign(e - s) * self.Ib_ramping_speed[0] * self.tp_interval))
        for Ib in tqdm(zip(*Ib_values), total=Ib_values[0].shape[0]):
            self.Ib = np.array(Ib)
            self.wait(seconds=self.tp_interval - self.t[-1])
            self.C[self.tes_flag] = self.Ce * (2.43 - 1.43 * self.Rt(self.T[0, self.tes_flag]) / self.Rt0)
            self.trigger(er=0, tpa=self.tpa_queue[self.tpa_idx],
                         verb=False, store=True, time_passes=True)
            self.tpa_idx += 1
            if self.tpa_idx + 1 > len(self.tpa_queue):
                self.tpa_idx = 0

    def send_testpulses(self, nmbr_tp=1):
        """
        TODO
        """
        for i in trange(nmbr_tp):
            self.wait(seconds=self.tp_interval - self.t[-1])
            self.C[self.tes_flag] = self.Ce * (2.43 - 1.43 * self.Rt(self.T[0, self.tes_flag]) / self.Rt0)
            self.trigger(er=0, tpa=self.tpa_queue[self.tpa_idx],
                         verb=False, store=True, time_passes=True)
            self.tpa_idx += 1
            if self.tpa_idx + 1 > len(self.tpa_queue):
                self.tpa_idx = 0

    def start_server(self, ):
        """
        TODO
        """
        pass  # TODO

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

    def plot_event(self, tes_channel=0, heater_channel=0, show=True):  # TODO make work with more components/tes/heaters
        """
        TODO
        """

        t_ch = np.arange(self.nmbr_components)[self.tes_flag][tes_channel]
        h_ch = np.arange(self.nmbr_components)[self.heater_flag][heater_channel]

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))

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

        axes[0, 1].plot(self.t, self.squid_out_noise, label='Squid output', zorder=5, c='black', linewidth=1, alpha=0.7)
        axes[0, 1].plot(self.t, self.squid_out, label='Recoil signature', zorder=10, c='red', linewidth=2, alpha=1)
        axes[0, 1].set_ylabel('Voltage (V)')  # , color='red'
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].legend(loc='upper right').set_zorder(100)
        axes[0, 1].set_zorder(10)

        Tmin, Tmax = np.min(self.T[:, t_ch]), np.max(self.T[:, t_ch])
        Rmin, Rmax = self.Rt(Tmin), self.Rt(Tmax)
        temp = np.linspace(np.minimum(self.Tc - 4 / self.k, np.minimum(Tmin, self.Tb(0))),
                           np.maximum(self.Tc + 4 / self.k, Tmax), 100)
        axes[1, 0].plot(temp, 1000 * self.Rt(temp), label='Transition curve', c='#FF7979', linewidth=2)
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

    def plot_nps(self, channel=0):
        """
        TODO
        """

        t_ch = np.arange(self.nmbr_components)[self.tes_flag][channel]
        Tt = self.T[0, t_ch]
        It = self.It[0, t_ch]

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        w, nps = self.get_nps(Tt, It, t_ch)
        ax.loglog(w, 1e6 * np.sqrt(nps), label='combined', linewidth=2, color='black', zorder=10)

        w = np.logspace(np.log10(np.min(w[w > 0])), np.log10(np.max(w)))
        nps = self.thermal_noise(w, Tt, It, t_ch)
        ax.loglog(w, 1e6 * np.sqrt(nps), label='Thermal noise', linewidth=2)

        nps[w > 0] = self.thermometer_johnson(w, Tt, It, t_ch)
        ax.loglog(w, 1e6 * np.sqrt(nps), label='Thermometer johnson', linewidth=2)

        nps = self.shunt_johnson(w, Tt, It, t_ch)
        ax.loglog(w, 1e6 * np.sqrt(nps), label='Shunt johnson', linewidth=2)

        nps = self.squid_noise(w, Tt, It, t_ch)
        ax.loglog(w, 1e6 * np.sqrt(nps), label='Squid noise', linewidth=2)

        nps = self.one_f_noise(w, Tt, It, t_ch)
        ax.loglog(w, 1e6 * np.sqrt(nps), label='1/f noise', linewidth=2)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (pA / sqrt(Hz))')
        ax.legend()

        fig.tight_layout()
        plt.show()

    def print_noise_parameters(self, channel=0):
        """
        TODO
        """

        t_ch = np.arange(self.nmbr_components)[self.tes_flag][channel]
        Tt = self.T[0, t_ch]
        It = self.It[0, t_ch]

        print('Resistance TES / Resistance normal conducting: {}'.format(self.Rt(Tt) / self.Rt0))
        print('Temperature mixing chamber: {} mK'.format(self.Tb(0)))
        print('Temperature TES: {} mK'.format(Tt))
        print('Resistance TES: {} mOhm'.format(1e3 * self.Rt(Tt)))
        print('Tau eff: {} ms'.format(1e3 * self.tau_eff(Tt, It)))
        print('Slope: {} mOhm/mK'.format(1e3 * self.dRtdT(Tt)))
        print('C: {} fJ / K '.format(1e6 * self.C[t_ch]))  # should be Pantic 3.5, and cubic
        print('Geff: ???')
        print('Tau in: {} ms'.format(1e3 * self.tau_in(channel)))
        print('Geb: {} pW / K '.format(1e3 * self.Gb[0]))
        print('G ETF: {} pW / K '.format(1e3 * self.G_etf(Tt, It, channel)))
        print('R shunt: {} mOhm'.format(1e3 * self.Rs))
        print('Temperature shunt: {} mK'.format(self.Tb(0)))
        print('i sq: {} pA/sqrt(Hz)'.format(1e12 * self.i_sq))
        print('1 / f amplitude: {} '.format(self.tes_fluct ** 2))

    def plot_buffer(self, tes_channel=0, tpa=None):
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
        TODO
        """
        T = 11.  # temp bath
        return T

    def P(self, t, T, It, no_pulses=False):
        """
        TODO
        """
        keV_to_pJ = e * 1e3 * 1e12
        if t > self.t0 and not no_pulses:
            P = self.er * self.eps * np.exp(
                -(t - self.t0) / self.lamb) / self.lamb * keV_to_pJ  # particle
        else:
            P = np.zeros(T.shape)
        if self.pileup_t0 is not None and t > self.pileup_t0 and not no_pulses:
            P += self.pileup_er * self.eps * np.exp(
                -(t - self.pileup_t0) / self.lamb) / self.lamb * keV_to_pJ  # particle
        P[self.tes_flag] += self.Rt(T[self.tes_flag]) * It ** 2  # self heating
        P += self.delta * self.heater_attenuator * self.dac / self.Rh  # heating
        if t > self.t0 and not no_pulses:
            P += np.maximum(self.tpa, 0) * self.delta * self.heater_attenuator * self.pulser_scale * np.exp(
                -(t - self.t0) / self.lamb) / self.Rh  # test pulses
        return P

    # private

    @staticmethod
    def dTdItdt(t, TIt, C, Gb, Tb, G, P, Rs, Ib, Rt, L, tes_flag):
        """
        TODO
        """
        nmbr_components = C.shape[0]
        dTdIt = np.zeros(nmbr_components + Ib.shape[0])
        T = TIt[:nmbr_components]
        It = TIt[nmbr_components:]

        dTdIt[:nmbr_components] = P(t, T, It)  # heat input
        dTdIt[:nmbr_components] += Gb * (Tb(t) - T)  # coupling to temperature bath
        dTdIt[:nmbr_components] += np.dot(G, T)  # heat transfer from other components
        dTdIt[:nmbr_components] -= np.dot(np.diag(np.dot(G, np.ones(T.shape[0]))),
                                          T)  # heat transfer to other components
        dTdIt[:nmbr_components] /= C  # heat to temperature

        dTdIt[nmbr_components:] = Rs * Ib  #
        dTdIt[nmbr_components:] -= It * (Rt(T[tes_flag]) + Rs)  #
        dTdIt[nmbr_components:] /= L  # voltage to current

        return dTdIt

    def Rt_init(self, a=1, b=1):
        """
        TODO
        """
        T = np.linspace(self.Tc - 10 / self.k, self.Tc + 10 / self.k, 500)
        R = np.zeros(T.shape)
        R[T < self.Tc] += a * self.Rt0 / (1 + np.exp(-self.k * (T[T < self.Tc] - self.Tc)))
        R[T >= self.Tc] += b * self.Rt0 / (1 + np.exp(-self.k * (T[T > self.Tc] - self.Tc)))
        R[np.logical_and(T >= self.Tc - 2 / self.k, T < self.Tc)] += (1 - a) * self.Rt0 * (
                T[np.logical_and(T >= self.Tc - 2 / self.k, T < self.Tc)] - self.Tc + 2 / self.k) * self.k / 4
        R[np.logical_and(T >= self.Tc, T < self.Tc + 2 / self.k)] += (1 - b) * self.Rt0 * (
                T[np.logical_and(T >= self.Tc, T < self.Tc + 2 / self.k)] - self.Tc + 2 / self.k) * self.k / 4
        R[T >= self.Tc + 2 / self.k] += (1 - b) * self.Rt0

        # make edges nicer
        idx_low = np.searchsorted(T, self.Tc - 3 / self.k)
        idx_high = np.searchsorted(T, self.Tc + 3 / self.k)
        R -= R[idx_low]
        R[R < 0] = 0
        R /= R[idx_high]
        R *= self.Rt0
        R[R > self.Rt0] = self.Rt0

        return lambda x: np.interp(x, T, R, left=0, right=self.Rt0)

    @staticmethod
    def pileup_er_distribution():
        """
        TODO
        """
        return np.random.exponential(10)  # in keV

    def currents(self, Tt):  # TODO make work with more components/tes/heaters
        """
        TODO
        """
        Rs = self.Rs
        Il, It = np.zeros(Tt.shape), np.zeros(Tt.shape)
        R_tes = self.Rt(Tt)
        Il[R_tes > 0] = self.Ib * (1 / (1 + Rs / R_tes[R_tes > 0]))
        It[R_tes > 0] = self.Ib * (1 / (1 + R_tes[R_tes > 0] / Rs))
        It[R_tes <= 0] = self.Ib
        return Il, It

    # noise contributions

    def get_noise_bl(self, channel=0, lamb=0.1):
        """
        TODO
        return in muA
        """

        t_ch = np.arange(self.nmbr_components)[self.tes_flag][channel]
        Tt = self.T[0, t_ch]  # in mK
        It = self.It[0, t_ch]  # in muA

        w, nps = self.get_nps(Tt, It, channel)

        a = np.sqrt(lamb)

        repeat = np.random.poisson(lam=1 / lamb)
        if repeat == 0:
            repeat = 1
        roll_values = np.random.randint(0, self.record_length, size=repeat)

        noise_temps = self.noise_function(nps, size=repeat)

        noise_temps[:] = np.roll(noise_temps, roll_values, axis=1)
        noise_temps[:] *= np.random.normal(scale=a, size=(repeat, 1))

        noise = np.sum(noise_temps, axis=0)

        return noise

    @staticmethod
    def noise_function(nps, size):
        """
        TODO
        return in muA
        """
        f = np.sqrt(nps)
        Np = (len(f) - 1) // 2
        f = np.array(f, dtype='complex')  # create array for frequencies
        f = np.tile(f, (size, 1))
        phases = np.random.rand(size, Np) * 2 * np.pi  # create random phases
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[:, 1:Np + 1] *= phases
        f[:, -1:-1 - Np:-1] = np.conj(f[:, 1:Np + 1])
        return np.fft.irfft(f, axis=-1)

    def get_nps(self, Tt, It, channel=0):
        """
        TODO
        inputs in mK and muA
        return in (muA)^2/Hz
        """
        w = np.fft.rfftfreq(self.record_length, 1 / self.sample_frequency)
        nps = np.zeros(w.shape)
        nps[w > 0] = self.thermal_noise(w[w > 0], Tt, It, channel) + \
                     self.thermometer_johnson(w[w > 0], Tt, It, channel) + \
                     self.shunt_johnson(w[w > 0], Tt, It, channel) + \
                     self.squid_noise(w[w > 0], Tt, It, channel) + \
                     self.one_f_noise(w[w > 0], Tt, It, channel)
        if self.lowpass is not None:
            b, a = butter(N=1, Wn=self.lowpass, btype='lowpass', analog=True)
            _, h = freqs(b, a, worN=w[w > 0])
            nps[w > 0] *= np.abs(h)
        return w, nps

    def dRtdT(self, T, channel=0):
        """
        TODO
        """
        grid = np.linspace(self.Tc - 5 / self.k, self.Tc + 5 / self.k, 1000)
        h = (grid[1] - grid[0])
        return np.interp(T, grid[:-1] + h / 2, 1 / h * np.diff(self.Rt(grid)))

    def G_etf(self, Tt, It, channel=0):
        """
        TODO
        """
        return It ** 2 * self.dRtdT(Tt, channel) * (self.Rt(Tt) - self.Rs) / (self.Rt(Tt) + self.Rs)

    def tau_eff(self, Tt, It, channel=0):
        """
        TODO
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][channel]
        return self.tau_in() / (1 + self.G_etf(Tt, It) / self.Gb[t_ch])

    def tau_in(self, channel=0):
        """
        TODO
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][channel]
        return self.C[t_ch] / self.Gb[t_ch]

    def thermal_noise(self, w, Tt, It, channel=0):
        """
        TODO
        return in (muA)^2/Hz
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][channel]
        P_th_squared = 4 * k * Tt ** 2 * self.Gb[t_ch]
        if Tt > self.Tb(0):
            P_th_squared *= 2 / 5 * (1 - (self.Tb(0) / Tt) ** 5) / (1 - (self.Tb(0) / Tt) ** 2)
        S_squared = 1 / (1 + w ** 2 * self.tau_eff(Tt, It, channel) ** 2)
        S_squared /= (self.Gb[t_ch] + self.G_etf(Tt, It, channel)) ** 2
        S_squared *= (It / (self.Rt(Tt) + self.Rs)) ** 2
        S_squared *= self.dRtdT(Tt, channel) ** 2
        return S_squared * P_th_squared * 1e6

    def thermometer_johnson(self, w, Tt, It, channel=0):
        """
        TODO
        return in (muA)^2/Hz
        """
        I_jt = (1 + w ** 2 * self.tau_in() ** 2)
        I_jt /= (1 + w ** 2 * self.tau_eff(Tt, It, channel) ** 2)
        I_jt *= 4 * k * Tt * self.Rt(Tt) / (self.Rt(Tt) + self.Rs) ** 2
        I_jt *= (self.tau_eff(Tt, It, channel) / self.tau_in()) ** 2
        return I_jt * 1e9

    def shunt_johnson(self, w, Tt, It, channel=0):
        """
        TODO
        return in (muA)^2/Hz
        """
        t_ch = np.arange(self.nmbr_components)[self.tes_flag][channel]
        I_js = (1 - It ** 2 / self.Gb[t_ch] * self.dRtdT(Tt, channel)) ** 2 + w ** 2 * self.tau_in() ** 2
        I_js /= (1 + w ** 2 * self.tau_eff(Tt, It, channel) ** 2)
        I_js *= 4 * k * Tt * self.Rs / (self.Rt(Tt) + self.Rs) ** 2
        I_js *= (self.tau_eff(Tt, It, channel) / self.tau_in()) ** 2
        return I_js * 1e9

    def squid_noise(self, w, Tt, It, channel=0):
        """
        TODO
        return in (muA)^2/Hz
        """
        I_sq = w / w
        I_sq *= self.i_sq[channel] ** 2
        return I_sq * 1e12

    def one_f_noise(self, w, Tt, It, channel=0):
        """
        TODO
        return in (muA)^2/Hz
        """
        I_one_f = (1 + w ** 2 * self.tau_in() ** 2)
        I_one_f /= (1 + w ** 2 * self.tau_eff(Tt, It, channel) ** 2)
        I_one_f *= (self.tau_eff(Tt, It, channel) / self.tau_in()) ** 2
        I_one_f *= It ** 2 * self.Rt(Tt) ** 2 * self.tes_fluct[channel] ** 2 / w / (self.Rt(Tt) + self.Rs) ** 2
        return I_one_f

    # utils

    def calc_par(self):  # TODO make work with more components/tes/heaters
        """
        TODO
        """
        self.offset = np.mean(self.squid_out_noise[self.t < self.t0], axis=0)
        self.ph = np.max(self.squid_out_noise - self.offset, axis=0)
        self.rms = np.std(self.squid_out_noise[self.t < self.t0] - self.offset, axis=0)

    def calc_par_(self):  # TODO make work with more components/tes/heaters
        """
        TODO
        """
        self.offset = np.mean(self.Il[:, self.t < self.t0], axis=1)
        self.ph = np.max(self.Il - self.offset, axis=1)
        self.rms = np.std(self.Il[:, self.t < self.t0], axis=1)

    def append_buffer(self):
        """
        TODO
        Attention, the dac and tes channels are wrongly aligned! (problem for more components)
        """
        self.buffer_offset.extend(self.offset)  # scalar
        self.buffer_ph.extend(self.ph)  # scalar
        self.buffer_rms.extend(self.rms)  # scalar
        self.buffer_dac.extend(self.dac)
        self.buffer_Ib.extend(self.Ib)
        self.buffer_tpa.extend(self.tpa * np.ones(self.nmbr_tes))  # scalar
        self.buffer_timer.extend(self.timer * np.ones(self.nmbr_tes))  # scalar
        self.buffer_channel.extend(np.arange(self.nmbr_tes))
        self.buffer_tes_resistance.extend(self.Rt(self.T[0, 0]) / self.Rt0 * np.ones(self.nmbr_tes))
        pulse = self.squid_out_noise.reshape(256, -1)
        pulse = np.mean(pulse, axis=-1)
        if self.store_raw:
            self.buffer_pulses.append(pulse)

        for ls in [self.buffer_offset, self.buffer_ph, self.buffer_rms,
                   self.buffer_dac, self.buffer_Ib, self.buffer_tpa,
                   self.buffer_timer, self.buffer_channel, self.buffer_tes_resistance,
                   self.buffer_pulses]:
            while len(ls) > self.max_buffer_len:
                del ls[0]
