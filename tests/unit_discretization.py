import numpy as np
from cryoenv.envs._discretization import action_to_discrete, \
    action_from_discrete, observation_from_discrete, \
    observation_to_discrete

V_SET_IV = (0, 99)
V_SET_STEP = 1
PH_IV = (0, 0.99)
PH_STEP = 0.01
WAIT_IV = (2, 100)
WAIT_STEP = 2
NMBR_CHANNELS = 1

nmbr_discrete_V = (V_SET_IV[1] - V_SET_IV[0]) / V_SET_STEP + 1
assert nmbr_discrete_V == np.floor(nmbr_discrete_V), 'nmbr_discrete_V must be whole number.'
nmbr_discrete_V = int(nmbr_discrete_V)
nmbr_discrete_ph = (PH_IV[1] - PH_IV[0]) / PH_STEP + 1
assert nmbr_discrete_ph == np.floor(nmbr_discrete_ph), 'nmbr_discrete_ph must be whole number.'
nmbr_discrete_ph = int(nmbr_discrete_ph)
nmbr_discrete_wait = (WAIT_IV[1] - WAIT_IV[0]) / WAIT_STEP + 1
assert nmbr_discrete_wait == np.floor(nmbr_discrete_wait), 'nmbr_discrete_wait must be whole number.'
nmbr_discrete_wait = int(nmbr_discrete_wait)
print('Nmbr discrete V: ', nmbr_discrete_V)
print('Nmbr discrete PH: ', nmbr_discrete_ph)
print('Nmbr discrete Wait: ', nmbr_discrete_wait)

used_actions = []

# actions grid
discrete_actions = np.empty((2, nmbr_discrete_V, nmbr_discrete_wait), dtype=int)
for i, resets in enumerate([True, False]):
    resets = np.array([resets], dtype=bool)
    for j, V_dec in enumerate(np.arange(V_SET_IV[0], V_SET_IV[1] + V_SET_STEP, V_SET_STEP)):
        V_dec = np.array([V_dec], dtype=float)
        for k, wait in enumerate(np.arange(WAIT_IV[0], WAIT_IV[1] + WAIT_STEP, WAIT_STEP)):
            wait = np.array(wait, dtype=float)
            discrete_actions[i, j, k] = action_to_discrete(resets, V_dec, wait,
                                                           wait_iv=WAIT_IV, V_iv=V_SET_IV, wait_step=WAIT_STEP,
                                                           V_step=V_SET_STEP)
            r, dV, w = action_from_discrete(n=discrete_actions[i, j, k], nmbr_channels=NMBR_CHANNELS,
                                            wait_iv=WAIT_IV, V_iv=V_SET_IV, wait_step=WAIT_STEP,
                                            V_step=V_SET_STEP)
            assert r == resets, f'Resets disagree: origin {resets}, conv {r}'
            if not r:
                assert dV == V_dec, f'Voltage decrease disagree: origin {V_dec}, conv {dV}'
            assert w == wait, f'Wait disagree: origin {wait}, conv {w}'

print(discrete_actions)

# observations grid
discrete_observations = np.empty((nmbr_discrete_V, nmbr_discrete_ph), dtype=int)
for i, V_set in enumerate(np.arange(V_SET_IV[0], V_SET_IV[1] + V_SET_STEP, V_SET_STEP)):
    V_set = np.array([V_set])
    for j, ph in enumerate(np.arange(PH_IV[0], PH_IV[1] + PH_STEP, PH_STEP)):
        ph = np.array([ph])
        discrete_observations[i, j] = observation_to_discrete(V_set, ph, V_iv=V_SET_IV, ph_iv=PH_IV, V_step=V_SET_STEP,
                                                              ph_step=PH_STEP)
        V, p = observation_from_discrete(discrete_observations[i, j], nmbr_channels=NMBR_CHANNELS, V_iv=V_SET_IV, ph_iv=PH_IV,
                                         V_step=V_SET_STEP, ph_step=PH_STEP)
        assert V == V_set, f'Voltage setpoints disagree: origin {V_set}, conv {V}'
        assert p == ph, f'Pulse height disagree: origin {ph}, conv {p}'

print(discrete_observations)
