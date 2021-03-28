# imports
import gym
import numpy as np
from itertools import product

gym.logger.set_level(40)

print('Create Environment.')
env = gym.make('cryoenv:cryoenv-v0', action_low=np.array([[0., 1., 0.]]))

for (i, j) in product(np.arange(0, 1, 0.1), repeat=2):
    print(i, j, env.temperature_model(P_R=np.array([i]),
                                      P_E=np.array([j])))
