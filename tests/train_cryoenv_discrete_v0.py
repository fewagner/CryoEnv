# %% markdown
# ## Introduction to Q-learning with OpenAI Gym
# This is a step-by-step guide to using Q-learning in a simple OpenAI gym environment
# %% markdown
# ### Table of Contents
#
# #### [Setup and Environment](#Setup_and_Environment)
# - in this section, we download and examine the environment after importing all the necessary libraries;
#
# #### [Q-learning](#Q-learning)
# - in this section, we use Q-learning to solve the Taxi problem.
#
# %% markdown
# ## Setup and Environment <a name='Setup_and_Environment'></a>
# %% markdown
# 1. install the necessary packages and libraries;
# 2. set up the Taxi environment;
# 3. determine the state and action space for our Q-table.
# %% codecell
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm.auto import tqdm, trange
# from IPython.display import clear_output

import warnings

warnings.simplefilter('ignore')

# %matplotlib inline
# import seaborn as sns
# sns.set()
# %% codecell
# Fixing seed for reproducibility
np.random.seed(0)
# %% codecell
# Loading and rendering the gym environment

gym.logger.set_level(40)

# constants
V_SET_IV = (0, 99)
V_SET_STEP = 1
PH_IV = (0, 0.99)
PH_STEP = 0.01
WAIT_IV = (10, 50)
WAIT_STEP = 2
HEATER_RESISTANCE = np.array([100.])
THERMAL_LINK_CHANNELS = np.array([[1.]])
THERMAL_LINK_HEATBATH = np.array([1.])
TEMPERATURE_HEATBATH = 0.
MIN_PH = 0.35
G = np.array([0.001])
T_HYST = np.array([0.001])
CONTROL_PULSE_AMPLITUDE = 50
ENV_FLUCTUATIONS = 0.005
k = np.array([15])
T0 = np.array([0.5])

print('Create Environment.')
env = gym.make('cryoenv:cryoenv-discrete-v0',
               V_set_iv=V_SET_IV,
               V_set_step=V_SET_STEP,
               ph_iv=PH_IV,
               ph_step=PH_STEP,
               wait_iv=WAIT_IV,
               wait_step=WAIT_STEP,
               heater_resistance=HEATER_RESISTANCE,
               thermal_link_channels=THERMAL_LINK_CHANNELS,
               thermal_link_heatbath=THERMAL_LINK_HEATBATH,
               temperature_heatbath=TEMPERATURE_HEATBATH,
               min_ph=MIN_PH,
               g=G,
               T_hyst=T_HYST,
               control_pulse_amplitude=CONTROL_PULSE_AMPLITUDE,
               env_fluctuations=ENV_FLUCTUATIONS,
               save_trajectory=True,
               k=k,
               T0=T0,
               )

# print('Check Environment.')
# check_env(env)


# %% codecell
# Getting the state space
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
# %% markdown
# ## Q-learning <a name='Q-learning'></a>
# %% markdown
# 1. initialize our Q-table given the state and action space in STEP 1;
#     - choose the hyperparameters for training;
# 2. choose an action: explore or exploit in STEP 2;
# 3. perform the action and measure the reward in STEPs 3 & 4;
# 4. ^^
# 5. update the Q-table using the Bellman equation in STEP 5.
#     - update the collected rewards
#     - use decay to balance exploration and exploitation
# %% codecell
# STEP 1 - Initializing the Q-table

# %% codecell
# Setting the hyperparameters

nmbr_agents = 1
train_episodes = 200
test_episodes = 1
max_steps = 5000

alpha = 0.7  # learning rate
discount_factor = 0.618
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 1 / train_episodes

plot_trajectories = [0, int(train_episodes / 2), train_episodes - 1]  # only for the first agent

# %% codecell
# Training the agent

# Creating lists to keep track of reward and epsilon values
training_rewards = np.empty((nmbr_agents, train_episodes), dtype=float)
epsilons = np.empty((nmbr_agents, train_episodes), dtype=float)
trajectories = np.empty((len(plot_trajectories), max_steps), dtype=float)

for agent in range(nmbr_agents):  # pool
    print('\nTrain Agent Nmbr: ', agent)
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
    print('Shape Q table: ', Q.shape)
    for episode in trange(train_episodes):
        # Reseting the environment each time as per requirement
        state = env.reset()
        # Starting the tracker for the rewards
        total_training_rewards = 0

        for step in range(max_steps):
            # Choosing an action given the states based on a random number
            exp_exp_tradeoff = random.uniform(0, 1)

            ### STEP 2: SECOND option for choosing the initial action - exploit
            # If the random number is larger than epsilon: employing exploitation
            # and selecting best action
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state, :])

            ### STEP 2: FIRST option for choosing the initial action - explore
            # Otherwise, employing exploration: choosing a random action
            else:
                action = env.action_space.sample()

            ### STEPs 3 & 4: performing the action and getting the reward
            # Taking the action and getting the reward and outcome state
            new_state, reward, done, info = env.step(action)

            ### STEP 5: update the Q-table
            # Updating the Q-table using the Bellman equation
            Q[state, action] = Q[state, action] + alpha * (
                    reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])
            # Increasing our total reward and updating the state
            total_training_rewards += reward
            state = new_state

            # Ending the episode
            if done == True:
                # print ("Total reward for episode {}: {}".format(episode, total_training_rewards))
                break

        # Cutting down on exploration by reducing the epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        # Adding the total reward and reduced epsilon values
        training_rewards[agent, episode] = total_training_rewards/max_steps
        epsilons[agent, episode] = epsilon

        # save the v set trajectory
        if agent == 0 and episode in plot_trajectories:
            print('Write trajectory of episode {} in index  {}.'.format(episode, plot_trajectories.index(episode)))
            trajectories[plot_trajectories.index(episode), :] = env.get_trajectory()[4][:, 0]  # TODO

# saving the Q table
np.save("q_table", Q)

# print("Training score over time: " + str(sum(training_rewards) / train_episodes))
# %% codecell
# Visualizing results and total reward over all episodes

fig, host = plt.subplots(figsize=(8, 5))

x = range(train_episodes)

par1 = host.twinx()

host.set_xlabel("Episode")
host.set_ylabel("Average Reward")
par1.set_ylabel("Epsilon")

color1 = 'green'
color2 = 'grey'

p1, = host.plot(x, np.mean(training_rewards, axis=0), color=color1, label="Reward", linewidth=2, zorder=15)
up = np.mean(training_rewards, axis=0) + np.std(training_rewards, axis=0)
low = np.mean(training_rewards, axis=0) - np.std(training_rewards, axis=0)
_ = host.fill_between(x, y1=up, y2=low, color=color1, zorder=5, alpha=0.3)
_ = host.text(x=0.5, y=0.95, s=r'$\alpha$: {}'.format(alpha), transform=host.transAxes, ha='center')
_ = host.text(x=0.5, y=0.9, s=r'$\gamma$: {}'.format(discount_factor), transform=host.transAxes, ha='center')
p2, = par1.plot(x, np.mean(epsilons, axis=0), color=color2, label="Epsilon", linewidth=2.5, alpha=0.5,
                linestyle='dashed', zorder=10)

par1.spines['right'].set_position(('outward', 0))

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

fig.tight_layout()
fig.suptitle('CryoEnvDiscrete v0: Q-Learning Training')
plt.savefig(fname='training.pdf')

plt.show()

# plot the voltage set point trajectory
plt.close()
for traj, nmbr in zip(trajectories, plot_trajectories):
    plt.plot(traj, linewidth=2, label='Trajectory ' + str(nmbr))
plt.title('Voltage Set Point Trajectories')
plt.xlabel('Environment Step')
plt.ylabel('Voltage Setpoint (a.u.)')
plt.legend()
plt.show()