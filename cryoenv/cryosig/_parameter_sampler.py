import numpy as np

def sample_parameters(**kwargs):
    def Rt(k, Tc, Rt0, a, b):
        """
        TODO
        """
        T = np.linspace(Tc - 10 / k, Tc + 10 / k, 500)
        R = np.zeros(T.shape)
        R[T < Tc] += a * Rt0 / (1 + np.exp(-k * (T[T < Tc] - Tc)))
        R[T >= Tc] += b * Rt0 / (1 + np.exp(-k * (T[T > Tc] - Tc)))
        R[np.logical_and(T >= Tc - 2 / k, T < Tc)] += (1 - a) * Rt0 * (
                    T[np.logical_and(T >= Tc - 2 / k, T < Tc)] - Tc + 2 / k) * k / 4
        R[np.logical_and(T >= Tc, T < Tc + 2 / k)] += (1 - b) * Rt0 * (
                    T[np.logical_and(T >= Tc, T < Tc + 2 / k)] - Tc + 2 / k) * k / 4
        R[T >= Tc + 2 / k] += (1 - b) * Rt0

        # make edges nicer
        idx_low = np.searchsorted(T, Tc - 3 / k)
        idx_high = np.searchsorted(T, Tc + 3 / k)
        R -= R[idx_low]
        R[R < 0] = 0
        R /= R[idx_high]
        R *= Rt0
        R[R > Rt0] = Rt0

        return lambda x: np.interp(x, T, R, left=0, right=Rt0)

    k = np.random.uniform(2, 5)
    Tc = np.random.uniform(15, 20)
    Rt0 = np.random.uniform(0.15, 0.2)
    a, b = np.random.uniform(0, 1, size=2)

    eps = np.random.uniform(0.05, 0.15)
    delt = np.random.uniform(0.01, 0.03)

    Tb = Tc - np.random.uniform(4, 8)

    G = np.random.uniform(2e-4, 2e-3)
    Gb = np.random.uniform(1e-3, 1e-2, size=2)
    C0 = np.random.uniform(1e-5, 1e-4)
    C1 = np.random.uniform(1e-4, 1e-3)

    pars = {
        "record_length": 16384,
        "sample_frequency": 25000,
        "C": np.array([C0, C1]),  # pJ / mK
        "Gb": Gb,  # pW / mK
        "G": np.array([[0., G], [G, 0.], ]),  # heat cond between components, pW / mK
        "lamb": np.random.uniform(0.001, 0.005),  # thermalization time (s)
        "eps": np.array([eps, (1 - eps), ]),  # share thermalization in components
        "delta": np.array([delt, (1 - delt), ]),  # share thermalization in components
        "Rs": np.random.uniform(0.03, 0.04, size=[1]),  # Ohm
        "Rh": np.random.uniform(5., 15., size=[1]),  # Ohm
        "Rt0": Rt0,  # Ohm
        "L": np.array([3.5e-7]),  # H
        "k": k,  # 1/mK
        "Tc": Tc,  # mK
        "Ib": np.array([1.]),  # muA
        "dac": np.array([0.]),  # V
        "pulser_scale": 1.,  # np.random.uniform(0.2, 1, size=[1]),  # scale factor for tpa
        "heater_attenuator": 1.,  # is multiplied to dac and tpa
        "tes_flag": np.array([True, False], dtype=bool),  # which component is a tes
        "heater_flag": np.array([False, True], dtype=bool),  # which component has a heater
        "t0": .16,  # onset of the trigger, s
        "pileup_prob": np.random.uniform(0.01, 0.1),  # percent / record window
        "tpa_queue": np.array([0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ]),  # V^2
        "tp_interval": 5,  # s
        "max_buffer_len": 10000,
        "dac_ramping_speed": np.array([2e-3]),  # V / s
        "Ib_ramping_speed": np.array([5e-3]),  # muA / s
        "xi": 1,  # squid conversion current to voltage, V / muA
        "i_sq": np.array([2 * 1e-12]),  # squid noise, A / sqrt(Hz)
        "tes_fluct": np.random.uniform(1e-4, 5e-3, size=[1]),  # percent
        "lowpass": 1e4,  # Hz
        "Tb": lambda x: Tb,  # function that takes one positional argument t, returns Tb
        "Rt": Rt(k, Tc, Rt0, a, b),
        "dac_range": (0., 5.),  # V
        "Ib_range": (1e-1, 1e1),  # muA
        "adc_range": (-10., 10.),  # V
    }

    for k, v in zip(kwargs.keys(), kwargs.values()):
        if k in pars:
            pars[k] = v

    return pars