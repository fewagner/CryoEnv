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
        self.trajectory_idx = np.zeros(self.buffer_size, dtype=np.int)

    def store_transition(self, state, action, reward, next_state, terminal, trajectory_idx=None):
        idx = self.buffer_counter % self.buffer_size
        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        # due to this being an infinite horizon problem
        # we don't have a terminal state
        self.terminal_memory[idx] = terminal
        if trajectory_idx is not None:
            self.trajectory_idx[idx] = trajectory_idx

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
        terminals = self.terminal_memory[batch_idxs]

        return states, actions, rewards, next_states, terminals

    def sample_trajectories(self, batch_size, steps, burn_in=0):
        if self.buffer_counter < self.buffer_size:
            start_idxs = np.random.choice(self.buffer_counter - steps, size=batch_size)
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

    def __len__(self):
        return min(self.buffer_size, self.buffer_counter)
