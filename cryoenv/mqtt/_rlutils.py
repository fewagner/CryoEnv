import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from copy import deepcopy
import pdb


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

        plt.plot(x_axis, returns / steps)
        plt.ylabel('Average Return')
        plt.xlabel('Episodes')
        plt.title(title)
        plt.show()

    def average(self):
        return np.mean(
            np.array(self.returns)[np.array(self.steps) > 0] / np.array(self.steps)[np.array(self.steps) > 0])

    def get_data(self):
        return np.array(self.steps)[np.array(self.steps) > 0], np.array(
            np.array(self.returns)[np.array(self.steps) > 0])


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
    def __init__(self, buffer_size, input_shape, n_actions, memmap_loc=None, store_trajectory_idx=True):
        buffer_size = int(buffer_size)
        self.buffer_size = buffer_size

        self.store_trajectory_idx = store_trajectory_idx

        self.memmap_loc = memmap_loc

        if self.memmap_loc is not None:
            mode = 'r+' if os.path.isfile(memmap_loc + 'state_memory.npy') else 'w+'
            self.state_memory = np.memmap(memmap_loc + 'state_memory.npy', dtype=float,
                                          shape=(buffer_size, *input_shape), mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'next_state_memory.npy') else 'w+'
            self.next_state_memory = np.memmap(memmap_loc + 'next_state_memory.npy', dtype=float,
                                               shape=(buffer_size, *input_shape), mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'action_memory.npy') else 'w+'
            self.action_memory = np.memmap(memmap_loc + 'action_memory.npy', dtype=float,
                                           shape=(buffer_size, n_actions), mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'reward_memory.npy') else 'w+'
            self.reward_memory = np.memmap(memmap_loc + 'reward_memory.npy', dtype=float, shape=(buffer_size,),
                                           mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'terminal_memory.npy') else 'w+'
            self.terminal_memory = np.memmap(memmap_loc + 'terminal_memory.npy', dtype=float, shape=(buffer_size,),
                                             mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'buffer_total.npy') else 'w+'
            self.buffer_total = np.memmap(memmap_loc + 'buffer_total.npy', dtype=int, shape=(1,), mode=mode)
            mode = 'r+' if os.path.isfile(memmap_loc + 'buffer_counter.npy') else 'w+'
            self.buffer_counter = np.memmap(memmap_loc + 'buffer_counter.npy', dtype=int, shape=(1,), mode=mode)
            if self.store_trajectory_idx:
                mode = 'r+' if os.path.isfile(memmap_loc + 'trajectory_idx.npy') else 'w+'
                self.trajectory_idx = np.memmap(memmap_loc + 'trajectory_idx.npy', dtype=int, shape=(buffer_size,),
                                                mode=mode)
        else:
            self.state_memory = np.zeros((self.buffer_size, *input_shape))
            self.next_state_memory = np.zeros((self.buffer_size, *input_shape))
            self.action_memory = np.zeros((self.buffer_size, n_actions))
            self.reward_memory = np.zeros(self.buffer_size)
            self.terminal_memory = np.zeros(self.buffer_size, dtype=np.bool)
            self.buffer_total = 0
            self.buffer_counter = 0
            if self.store_trajectory_idx:
                self.trajectory_idx = np.zeros(self.buffer_size, dtype=np.int)

    def store_transition(self, state, action, reward, next_state, terminal, trajectory_idx=None):
        idx = self.buffer_counter  # % self.buffer_size
        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = terminal
        if trajectory_idx is not None:
            self.trajectory_idx[idx] = trajectory_idx

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

    def sample_trajectories(self, batch_size, steps, burn_in=0):
        buffer_counter = np.array(self.buffer_counter).reshape(1)[0]
        if self.buffer_total < self.buffer_size:
            start_idxs = np.random.choice(buffer_counter - steps, size=batch_size)
            idxs = start_idxs.reshape(-1, 1) * np.ones((batch_size, steps), dtype=int) + np.arange(steps,
                                                                                                   dtype=int).reshape(1,
                                                                                                                      -1)
        else:
            start_idxs = np.random.choice(self.buffer_size, batch_size)
            idxs = start_idxs.reshape(-1, 1) * np.ones((batch_size, steps), dtype=int) + np.arange(steps,
                                                                                                   dtype=int).reshape(1,
                                                                                                                      -1)
            idxs %= self.buffer_size

        states = self.state_memory[idxs]
        next_states = self.next_state_memory[idxs]
        actions = self.action_memory[idxs]
        rewards = self.reward_memory[idxs]
        terminals = self.terminal_memory[idxs]
        trajectory_idx = self.trajectory_idx[idxs]

        loss_mask = np.ones((batch_size, steps), dtype=bool)

        for i, diffs in enumerate(np.diff(trajectory_idx)):
            nonzeros = np.nonzero(diffs - 1)[0]
            if len(nonzeros) > 0:
                loss_mask[i, nonzeros[0] + 1:] = False
            else:
                pass

        loss_mask[:, :burn_in] = False

        return states, actions, rewards, next_states, terminals, trajectory_idx, loss_mask

    def erase(self):
        self.state_memory[:] = np.zeros(self.state_memory.shape)
        self.next_state_memory[:] = np.zeros(self.next_state_memory.shape)
        self.action_memory[:] = np.zeros(self.action_memory.shape)
        self.reward_memory[:] = np.zeros(self.reward_memory.shape)
        self.terminal_memory[:] = np.zeros(self.terminal_memory.shape)
        self.buffer_total[:] = np.zeros(self.buffer_total.shape)
        self.buffer_counter[:] = np.zeros(self.buffer_counter.shape)
        if self.store_trajectory_idx is not None:
            self.trajectory_idx[:] = np.zeros(self.trajectory_idx.shape)
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
            if self.store_trajectory_idx is not None:
                self.trajectory_idx.flush()

    def __len__(self):
        return min(self.buffer_size, self.buffer_counter)

# TODO offline trajectory buffer


def generate_sweep(nmbr_dac, nmbr_bias):
    sweep = []
    dacs = np.linspace(1, -1, nmbr_dac)
    bias = np.linspace(-1, 1, nmbr_bias)
    for i, d in enumerate(dacs):
        array = np.flip(bias) if i % 2 == 1 else bias
        sgn = +1 if i % 2 == 1 else -1
        for j, b in enumerate(array):
            sweep.append([d, b + sgn * j / (nmbr_bias - 1) / nmbr_bias])
    return np.array(sweep)


def augment_pars(pars, scale=0.1, **kwargs):
    new_pars = deepcopy(pars)

    for k, v in zip(kwargs.keys(), kwargs.values()):
        new_pars[k] = v

    nmbr_components = len(new_pars['C'])
    nmbr_tes = len(new_pars['Rt0'])
    nmbr_heater = len(new_pars['Rh'])

    for i in range(0, nmbr_components - 1):
        for j in range(i + 1, nmbr_components):
            new_pars['G'][i, j] = new_pars['G'][i, j] * (1 + scale * np.random.normal())
            new_pars['G'][j, i] = new_pars['G'][i, j]
            new_pars['G'] = new_pars['G'] * (1 + scale * np.random.normal())
    if nmbr_components == 2:
        new_pars['eps'][:, 0] = 1 / (
                (1 / new_pars['eps'][:, 0] - 1) * (1 + scale * np.random.normal(size=nmbr_components)) / (
                1 + scale * np.random.normal(size=nmbr_components)) + 1)
        new_pars['eps'][:, 1] = 1 - new_pars['eps'][:, 0]
    elif nmbr_components == 3:
        if nmbr_tes == 2:
            new_eps = np.zeros((3,3))
            for i in range(2):
                new_eps[i, i] = 1 / (
                        (1 / new_pars['eps'][0, 0] - 1) * (1 + scale * np.random.normal()) / (
                        1 + scale * np.random.normal()) + 1)
                new_eps[2, i] = 1 / (
                        (1 / new_pars['eps'][2, 0] - 1) * (1 + scale * np.random.normal()) / (
                        1 + scale * np.random.normal()) + 1)
            new_eps[:, 2] = 1 - np.sum(new_eps[:, :2], axis=1)
            new_pars['eps'] = new_eps
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    new_pars['Ib_range'] = (new_pars['Ib_range'][0] * (1 + scale * np.random.normal()),
                            new_pars['Ib_range'][1] * (1 + scale * np.random.normal()))
    new_pars['dac_range'] = (new_pars['dac_range'][0] * (1 + scale * np.random.normal()),
                             new_pars['dac_range'][1] * (1 + scale * np.random.normal()))
    for name in ['C', 'Gb', 'lamb']:
        new_pars[name] = new_pars[name] * (1 + scale * np.random.normal(size=nmbr_components))
    for name in ['Rs', 'L', 'Rt0', 'k', 'Tc', 'Ib', 'eta', 'i_sq', 'tes_fluct', 'flicker_slope',
                 'excess_johnson']:
        new_pars[name] = new_pars[name] * (1 + scale * np.random.normal(size=nmbr_tes))
    new_pars['emi'] = new_pars['emi'] * (1 + scale * np.random.normal(size=(nmbr_tes, 3)))
    for name in ['lamb_tp', 'Rh', 'dac', 'pulser_scale', 'heater_current', 'tau_cap']:
        new_pars[name] = new_pars[name] * (1 + scale * np.random.normal(size=nmbr_heater))
    if 'Tb' in kwargs:
        tb_temp = float(kwargs['Tb'](0)) * (1 + scale * np.random.normal())
        new_pars['Tb'] = lambda t: tb_temp
    elif 'Tb' in pars:
        tb_temp = float(pars['Tb'](0)) * (1 + scale * np.random.normal())
        new_pars['Tb'] = lambda t: tb_temp

    return new_pars


def double_tes(pars):
    pars_2tes = deepcopy(pars)

    pars_2tes['C'] = np.array([pars['C'][0] / 2, pars['C'][0] / 2, pars['C'][1]])
    pars_2tes['Gb'] = np.array([pars['Gb'][0] / 2, pars['Gb'][0] / 2, pars['C'][1]])
    pars_2tes['G'] = np.array([[0., 0., pars['G'][0, 1] / 2],
                               [0., 0., pars['G'][0, 1] / 2],
                               [pars['G'][0, 1] / 2, pars['G'][0, 1] / 2, 0.]])
    pars_2tes['lamb'] = np.array([pars['lamb'][0], pars['lamb'][0], pars['lamb'][1]])
    pars_2tes['lamb_tp'] = np.array([pars['lamb_tp'][0], pars['lamb_tp'][0]])
    pars_2tes['eps'] = np.array([[0.99, 0., 1 - 0.99],
                                 [0.99, 0., 1 - 0.99],
                                 [pars['eps'][1, 0] / 2, pars['eps'][1, 0] / 2, 1 - pars['eps'][1, 0] / 2], ])
    pars_2tes['Rs'] = np.array([pars['Rs'][0], pars['Rs'][0]])
    if pars['delta_h'][0,0] > 0.5:
        pars_2tes['Rh'] = np.array([pars['Rh'][0], pars['Rh'][0]])
    else:
        pars_2tes['Rh'] = np.array([pars['Rh'][0]/10, pars['Rh'][0]/10])
    pars_2tes['L'] = np.array([pars['L'][0], pars['L'][0]])
    pars_2tes['Rt0'] = np.array([pars['Rt0'][0], pars['Rt0'][0]])
    pars_2tes['k'] = np.array([pars['k'][0], pars['k'][0]])
    pars_2tes['Tc'] = np.array([pars['Tc'][0], pars['Tc'][0]])
    pars_2tes['Ib'] = np.array([pars['Ib'][0], pars['Ib'][0]])
    pars_2tes['dac'] = np.array([pars['dac'][0]/2, pars['dac'][0]/2])
    pars_2tes['pulser_scale'] = np.array([pars['pulser_scale'][0], pars['pulser_scale'][0]])
    pars_2tes['heater_current'] = np.array([pars['heater_current'][0], pars['heater_current'][0]])
    pars_2tes['tes_flag'] = [True, True, False]
    pars_2tes['heater_flag'] = [True, True, False]
    pars_2tes['i_sq'] = np.array([pars['i_sq'][0], pars['i_sq'][0]])
    pars_2tes['tes_fluct'] = np.array([pars['tes_fluct'][0], pars['tes_fluct'][0]])
    pars_2tes['flicker_slope'] = np.array([pars['flicker_slope'][0], pars['flicker_slope'][0]])
    pars_2tes['emi'] = np.array([pars['emi'][0], pars['emi'][0]])
    pars_2tes['tau_cap'] = np.array([pars['tau_cap'][0], pars['tau_cap'][0]])
    pars_2tes['excess_johnson'] = np.array([pars['excess_johnson'][0], pars['excess_johnson'][0]])
    pars_2tes['delta'] = np.array([[pars['delta'][0, 0] + (1 -pars['delta'][0, 0])/2, 0., 1 - pars['delta'][0, 0] - (1 -pars['delta'][0, 0])/2],
                                 [0., pars['delta'][0, 0] + (1 -pars['delta'][0, 0])/2, 1 - pars['delta'][0, 0] - (1 -pars['delta'][0, 0])/2], ])
    pars_2tes['delta_h'] = np.array([[pars['delta_h'][0, 0] + (1 - pars['delta_h'][0, 0]) / 2, 0.,
                                    1 - pars['delta_h'][0, 0] - (1 - pars['delta_h'][0, 0]) / 2],
                                   [0., pars['delta_h'][0, 0] + (1 - pars['delta_h'][0, 0]) / 2,
                                    1 - pars['delta_h'][0, 0] - (1 - pars['delta_h'][0, 0]) / 2], ])
    pars_2tes['Ib_ramping_speed'] = np.array([5e-3, 5e-3])
    pars_2tes['dac_ramping_speed'] = np.array([2e-3, 2e-3])
    pars_2tes['eta'] = np.array([pars['eta'][0], pars['eta'][0]])
    pars_2tes['excess_phonon'] = np.array([1., 1.])
    pars_2tes['pileup_comp'] = 2

    return pars_2tes
