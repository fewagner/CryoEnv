import numpy as np
from cryoenv.envs._discretization import action_to_discrete, action_from_discrete, observation_from_discrete, observation_to_discrete

reset = np.array([False, False, True])
dV = np.array([14, 85, 99])
wait = np.array(10)
nmbr_channels = 3

print('\nACTION TO DISCRETE')
discrete = action_to_discrete(reset, dV, wait)

print('\nACTION FROM DISCRETE')
from_discrete = action_from_discrete(discrete, nmbr_channels)


print('\nSUMMARY')
print('origin: ', reset, dV, wait)
print('conv: ', from_discrete)

V = np.array([0])
ph = np.array([0.29])
nmbr_channels = 1

print('\nOBSERVATION TO DISCRETE')
discrete = observation_to_discrete(V, ph)

print('\nOBSERVATION FROM DISCRETE')
from_discrete = observation_from_discrete(discrete, nmbr_channels)


print('\nSUMMARY')
print('origin: ', V, ph)
print('conv: ', from_discrete)
