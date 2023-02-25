import numpy as np

def Rt_smooth(k, Tc, Rt0, a=1, b=1):
    """
    TODO
    a and b make the curve less linear and more sigmoid
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

def Rt_kinky(k, Tc, Rt0, alpha=5, beta=1, length=1000):
    """
    TODO
    """
    length = int(np.maximum(np.random.normal(length,int(length/5)), int(length/10))) + 1
    alpha = np.random.poisson(alpha) + 1

    rvals = np.random.normal(size=length)
    curve = np.cumsum(rvals**2)
    curve /= curve[-1]
    curve **= alpha
    curve = 2/(1 + np.exp(-7*curve**beta)) - 1
    curve *= Rt0

    return lambda x: np.interp(x, np.arange(0,length)/length*4/k + Tc - 2/k, curve, left=0, right=Rt0)