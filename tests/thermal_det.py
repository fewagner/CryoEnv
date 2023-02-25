import numpy as np
from scipy.integrate import odeint, solve_ivp
import numba as nb
import matplotlib.pyplot as plt
import time
from numbalsoda import lsoda_sig, lsoda

# @nb.njit
def det(t, T, C, Gb, Tb, G, P, t0):
    dT = P(t, t0)  # heat input
    dT += Gb * (Tb - T)  # coupling to temperature bath
    dT += np.dot(G, T)  # heat transfer from other components
    dT -= np.dot(np.diag(np.dot(G, np.ones(T.shape[0]))), T)  # heat transfer to other components
    dT /= C  # heat to temperature

    return dT


# -------------------------------------------------------
# constants
# -------------------------------------------------------

C = np.array([0.1,
              10.,
              50.])  # heat capacity
Gb = np.array([100.,
               10.,
               100.]) # heat cond to bath

Tb = 0.  # temp bath
Gab = 10.
Gbc = 1.
G = np.array([[0., Gab, 0.],
              [Gab, 0., Gbc],
              [0., Gbc, 0.], ])  # heat cond between components

lamb = 0.01  # thermalization time
eps = 0.1  # share in thermometer
eps_ = np.array([eps,
                 2*eps,
                 (1-3*eps)])
t0 = 0.03

# @nb.njit
def P(x, t0):
    if t0 < x:
        return eps_*np.exp(-(x-t0)/lamb)
    else:
        return np.zeros(eps_.shape)


t = np.linspace(0,0.1,10000)
T0 = Tb * np.ones(C.shape[0])

# -------------------------------------------------------
# calculations
# -------------------------------------------------------

start = time.time()
T1 = odeint(det, T0, t, args=(C, Gb, Tb, G, P, t0), tfirst=True)
print('Calculation done in {} ms.'.format(time.time() - start))

start = time.time()
out = solve_ivp(det, (t[0], t[-1]), T0, t_eval=t, args=(C, Gb, Tb, G, P, t0), method='LSODA')
T2 = out.y.T
print('Calculation done in {} ms.'.format(time.time() - start))

# -------------------------------------------------------
# plots
# -------------------------------------------------------

plt.figure(dpi=150)
for i, (T, ls) in enumerate(zip([T1, T2], ['solid', 'dotted'])):
    plt.plot(t, T[:, 0], label='Thermometer' + str(i), c='C0', linewidth=2, linestyle=ls)
    plt.plot(t, T[:, 1], label='Carrier' + str(i), c='C1', linewidth=2, linestyle=ls)
    plt.plot(t, T[:, 2], label='Crystal' + str(i), c='C2', linewidth=2, linestyle=ls)
plt.legend()
plt.xlabel('Time (a.u.)')
plt.ylabel('Temperature (a.u.)')
plt.show()