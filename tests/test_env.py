# imports
import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
gym.logger.set_level(40)

print('Create Environment.')
env = gym.make('cryoenv:cryoenv-v0', save_trajectory=True)

print('Check Environment.')
check_env(env)

print('Create Model.')
model = A2C("MlpPolicy", env, verbose=False)

print('Learn.')
model.learn(total_timesteps=500)

print('Save the trajectory.')
actions, new_states, rewards = env.get_trajectory()
np.savetxt('data/actions.txt', actions)
np.savetxt('data/new_states.txt', new_states)
np.savetxt('data/rewards.txt', rewards)

print('Plot the rewards.')
plt.close()
plt.plot(rewards)
plt.show()