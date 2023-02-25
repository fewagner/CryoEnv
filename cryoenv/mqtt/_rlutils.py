import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

class ReturnTracker():
    
    def __init__(self):
        self.returns = []
        self.steps = []
        self.collected_rewards = 0
        self.step = 0
    
    def new_episode(self):
        if self.step > 0:
            self.returns.append(self.collected_rewards)
            self.steps.append(self.step)
            self.collected_rewards = 0
            self.step = 0
    
    def add(self, reward):
        self.collected_rewards += reward
        self.step += 1
    
    def plot(self, title=None, smooth=1):
        
        returns = np.array(self.returns)
        steps = np.array(self.steps)
        returns = returns[np.array(self.steps) > 0]
        steps = steps[np.array(self.steps) > 0]
        x_axis = np.arange(len(steps))
        
        if smooth > 1:
            cut = len(steps) - len(steps) % smooth
            x_axis = x_axis[:cut]
            steps = steps[:cut]
            returns = returns[:cut]
            x_axis = np.floor(np.mean(x_axis.reshape(-1, smooth), axis=1))
            steps = np.mean(steps.reshape(-1, smooth), axis=1)
            returns = np.mean(returns.reshape(-1, smooth), axis=1)
        
        plt.plot(x_axis, returns/steps)
        plt.ylabel('Average Return')
        plt.xlabel('Episodes')
        plt.title(title)
        plt.show()
    
    def average(self):
        return np.mean(np.array(self.returns)[np.array(self.steps) > 0]/np.array(self.steps)[np.array(self.steps) > 0])
    
    def get_data(self):
        return np.array(self.steps)[np.array(self.steps) > 0], np.array(np.array(self.returns)[np.array(self.steps) > 0])
    
class Agent():
    
    def __init__(self):
        raise NotImplementedError('Agent class requires that you implement __init__!')
    
    def learn(self):
        raise NotImplementedError('Agent class requires that you implement learn!')
    
    def predict(self, state):
        raise NotImplementedError('Agent class requires that you implement predict!')

    
class HistoryWriter:
    
    def __init__(self):
        self.erase()
    
    def add_scalar(self, name, value, step):
        if name not in self.history:
            self.history[name] = {}
        self.history[name][step] = value
        
    def erase(self):
        self.history = {}
        
    def plot(self, name):
        plt.scatter(self.history[name].keys(), self.history[name].values())
        plt.xlabel('step')
        plt.ylabel('value')
        plt.title(name)
        plt.show()

class ReplayBuffer:
    def __init__(self, buffer_size, input_shape, n_actions, memmap_loc=None):
        buffer_size = int(buffer_size)
        self.buffer_size = buffer_size

        self.memmap_loc = memmap_loc

        if self.memmap_loc is not None:
            mode = 'r+' if os.path.isfile(memmap_loc + 'state_memory.npy') else 'w+'
            self.state_memory = np.memmap(memmap_loc + 'state_memory.npy', dtype=float, shape=(buffer_size, *input_shape), mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'next_state_memory.npy') else 'w+'
            self.next_state_memory = np.memmap(memmap_loc + 'next_state_memory.npy', dtype=float, shape=(buffer_size, *input_shape), mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'action_memory.npy') else 'w+'
            self.action_memory = np.memmap(memmap_loc + 'action_memory.npy', dtype=float, shape=(buffer_size, n_actions), mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'reward_memory.npy') else 'w+'
            self.reward_memory = np.memmap(memmap_loc + 'reward_memory.npy', dtype=float, shape=(buffer_size,), mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'terminal_memory.npy') else 'w+'
            self.terminal_memory = np.memmap(memmap_loc + 'terminal_memory.npy', dtype=float, shape=(buffer_size,), mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'buffer_total.npy') else 'w+'
            self.buffer_total = np.memmap(memmap_loc + 'buffer_total.npy', dtype=int, shape=(1,), mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'buffer_counter.npy') else 'w+'
            self.buffer_counter = np.memmap(memmap_loc + 'buffer_counter.npy', dtype=int, shape=(1,), mode=mode)
        else:
            self.state_memory = np.zeros((self.buffer_size, *input_shape))
            self.next_state_memory = np.zeros((self.buffer_size, *input_shape))
            self.action_memory = np.zeros((self.buffer_size, n_actions))
            self.reward_memory = np.zeros(self.buffer_size)
            self.terminal_memory = np.zeros(self.buffer_size, dtype=np.bool)
            self.buffer_total = 0
            self.buffer_counter = 0

    def store_transition(self, state, action, reward, next_state, terminal):
        idx = self.buffer_counter  # % self.buffer_size
        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = terminal

        self.buffer_counter += 1
        self.buffer_counter %= self.buffer_size
        self.buffer_total += 1
        
        self.flush()

    def sample_buffer(self, batch_size):
        buffer_counter = np.array(self.buffer_counter).reshape(1)[0]
        if self.buffer_total < self.buffer_size:
            batch_idxs = np.random.choice(buffer_counter, batch_size)
        else:
            batch_idxs = np.random.choice(self.buffer_size, batch_size)

        states = self.state_memory[batch_idxs]
        next_states = self.next_state_memory[batch_idxs]
        actions = self.action_memory[batch_idxs]
        rewards = self.reward_memory[batch_idxs]
        terminals = self.terminal_memory[batch_idxs]

        return states, actions, rewards, next_states, terminals
    
    def erase(self):
        self.state_memory[:] = np.zeros(self.state_memory.shape)
        self.next_state_memory[:] = np.zeros(self.next_state_memory.shape)
        self.action_memory[:] = np.zeros(self.action_memory.shape)
        self.reward_memory[:] = np.zeros(self.reward_memory.shape)
        self.terminal_memory[:] = np.zeros(self.terminal_memory.shape)
        self.buffer_total[:] = np.zeros(self.buffer_total.shape)
        self.buffer_counter[:] = np.zeros(self.buffer_counter.shape)
        
        self.flush()
        
    def flush(self):
        if self.memmap_loc is not None:
            self.state_memory.flush()
            self.next_state_memory.flush()
            self.action_memory.flush()
            self.reward_memory.flush()
            self.terminal_memory.flush()
            self.buffer_total.flush()
            self.buffer_counter.flush()

    def __len__(self):
        return min(self.buffer_size, self.buffer_counter)

    
def generate_sweep(nmbr_dac,nmbr_bias):
    sweep = []
    dacs = np.linspace(1,-1,nmbr_dac)
    bias = np.linspace(-1,1,nmbr_bias)
    for i,d in enumerate(dacs):
        array = np.flip(bias) if i%2==1 else bias
        sgn = +1 if i%2==1 else -1
        for j,b in enumerate(array):
            sweep.append([d,b+sgn*j/(nmbr_bias-1)/nmbr_bias])
    return np.array(sweep)