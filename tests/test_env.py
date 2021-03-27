
# imports
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

print('Create Environment.')
env = gym.make('cryoenv:cryoenv-v0')

print('Check Environment.')
check_env(env)

print('Create Model.')
model = A2C("MlpPolicy", env, verbose=False)

print('Learn.')
model.learn(total_timesteps=10000)

print('Training.')
obs = env.reset()
for i in range(10):
    print('Step: ', i)
    action, _states = model.predict(obs)
    print('Action, States: ', action, _states)
    obs, rewards, dones, info = env.step(action)
    print('Obs, Rew, Done: ', obs, rewards, dones)
    env.render()
