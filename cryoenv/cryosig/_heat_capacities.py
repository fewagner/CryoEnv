import numpy as np

def tes_heat_capacity(k, Tc, Rt):

    T = np.linspace(Tc - 10 / k, Tc + 10 / k, 500)
    factor = (2.43 - 1.43 * Rt(T) / Rt(T[-1]))

    return lambda x: np.interp(x, T, factor, left=2.43, right=1)