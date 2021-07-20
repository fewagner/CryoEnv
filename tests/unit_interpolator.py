import numpy as np
import gym
from gym import error, spaces, utils
from cryoenv.agents._interpolator import Interpolator
import matplotlib.pyplot as plt


def dummy_reward(action: list, observation: list):
    return 1 - 1 / (len(action) + len(observation)) * (np.sum(action ** 2) + np.sum(observation ** 2))


nmbr_actions = 1
nmbr_observations = 1
nmbr_steps = 10000

action_space = spaces.Box(low=- np.ones(nmbr_actions),
                          high=np.ones(nmbr_actions),
                          dtype=np.float32)
observation_space = spaces.Box(low=- np.ones(nmbr_observations),
                               high=np.ones(nmbr_observations),
                               dtype=np.float32)

value_function = Interpolator(maxlen=1000, initval=0, allowed_overdensity=5, method='nearest')
value_function.define_spaces(action_space, observation_space)

# get random observation and action

for _ in range(nmbr_steps):
    action = action_space.sample()
    obs = observation_space.sample()
    reward = dummy_reward(action, obs)

    value_function.update(action, obs, reward)

# make grid for plotting

x_a = np.arange(-1, 1, 0.01)
x_o = np.arange(0, 1, 0.2)

plt.close()
for i, oval in enumerate(x_o):
    truth = [dummy_reward(np.array([a]), np.array([oval])) for a in x_a]
    approx = [value_function.predict([a], [oval]) for a in x_a]
    plt.plot(truth, linewidth=0.7, color='C' + str(i))
    plt.plot(approx, linestyle='dashed', linewidth=2, alpha=0.7, color='C' + str(i))
plt.show()
