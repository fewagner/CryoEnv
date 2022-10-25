import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, input_shape, n_actions):
        self.buffer_size = buffer_size
        self.buffer_counter = 0

        self.state_memory = np.zeros((self.buffer_size, *input_shape))
        self.next_state_memory = np.zeros((self.buffer_size, *input_shape))
        self.action_memory = np.zeros((self.buffer_size, n_actions))
        self.reward_memory = np.zeros(self.buffer_size)
        self.terminal_memory = np.zeros(self.buffer_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.buffer_counter % self.buffer_size
        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        # due to this being an infinite horizon problem
        # we don't have a terminal state
        self.terminal_memory[idx] = done

        self.buffer_counter += 1

    def sample_buffer(self, batch_size):
        if self.buffer_counter < self.buffer_size:
            batch_idxs = np.random.choice(self.buffer_counter, batch_size)
        else:
            batch_idxs = np.random.choice(self.buffer_size, batch_size)

        states = self.state_memory[batch_idxs]
        next_states = self.next_state_memory[batch_idxs]
        actions = self.action_memory[batch_idxs]
        rewards = self.reward_memory[batch_idxs]
        dones = self.terminal_memory[batch_idxs]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return min(self.buffer_size, self.buffer_counter)
