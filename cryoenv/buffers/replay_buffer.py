from typing import Tuple
import numpy as np


class ReplayBuffer:

    def __init__(self, max_size: int, input_shape: Tuple[int], n_actions: int) -> None:
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done) -> None:
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done  # we don't really have a terminal state

        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int) -> Tuple:
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)
