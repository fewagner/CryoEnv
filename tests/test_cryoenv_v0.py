# imports
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.env_checker import check_env
gym.logger.set_level(40)

print('Create Environment.')
env = gym.make('cryoenv:cryoenv-v0',
               save_trajectory=True,
               alpha=0.1,
               beta=0.1,
               gamma=1,
               )

print('Check Environment.')
check_env(env)

print('Create Model.')
model = A2C("MlpPolicy", env, verbose=False)

print('Learn.')
model.learn(total_timesteps=1000)

print('Save the trajectory.')
actions, new_states, rewards = env.get_trajectory()
np.savetxt('data/actions.txt', actions)
np.savetxt('data/new_states.txt', new_states)
np.savetxt('data/rewards.txt', rewards)

print('Plot the rewards.')
plt.close()
plt.plot(rewards)
plt.show()

print('Plot the sensor model.')
k, T0 = env.k, env.T0
print('k, T0: ', k, T0)
plt.close()
grid = np.arange(-0.5, 1.5, 0.01)
plt.plot(grid, env.sensor_model(grid, k, T0))
plt.axvline(x=0, color='black')
plt.axvline(x=1, color='black')
plt.show()