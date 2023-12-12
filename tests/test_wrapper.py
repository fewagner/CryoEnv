import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import warnings
from stable_baselines3.common.env_checker import check_env

warnings.simplefilter('ignore')
# np.random.seed(0)
gym.logger.set_level(40)

env = gym.make('cryoenv:cryoenv-sig-v0',
               omega=1e-5,
               sample_pars=True)

check_env(env)
observation = env.reset()
n_steps = 5

for i in range(n_steps):
    env.render(mode='human')
    # fig, axes = env.render(mode='mpl')
    # plt.savefig('data/img{}.png'.format(i))
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    print("{}. dac: {}, bias: {}, reward: {}".format(i, observation[1], observation[2], reward))

    if done:
        observation = env.reset()

env.close()
