import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from tqdm import tqdm


class SAC_v2:
    def __init__(self, env, state_dim, action_dim,
                 batch_size=256, lr_actor=3e-4, lr_crit_val=3e-4, init_weights: Optional[str] = None,
                 tau=0.005, reward_scale=1.0, buffer_max_size=1000000, hidden_dims = [256, 256],
                 gamma=0.99, target_update_interval=1
                 ):
        """
        Args:
            env: gym environment, used for training.
            state_dim: env.observation_space.shape[0]
            action_dim: env.action_space.shape[0]
            tau: target update factor
        """

        self.env = env
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_crit_val = lr_crit_val
        self.tau = tau
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.buffer = ReplayBuffer(buffer_max_size, state_dim, action_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = PolicyNetwork(state_dim, action_dim, action_scaler=self.env.action_space.high,
                                    lr=lr_actor, device=self.device)
        self.qf_1 = QNetwork(state_dim, action_dim, lr=lr_crit_val, name='qf_1', device=self.device)
        self.qf_2 = QNetwork(state_dim, action_dim, lr=lr_crit_val, name='qf_2', device=self.device)

        self.value = ValueNetwork(state_dim, lr=lr_crit_val, device=self.device)
        self.target_value = ValueNetwork(state_dim, lr=lr_crit_val, name='target_value', device=self.device)

        self.reward_scale = reward_scale

        if init_weights:
            # TODO: implement kaiming init, etc.
            pass

    def choose_action(self, state):
        state = torch.tensor(np.array([state])).float().to(self.device)
        action, _ = self.policy.sample_action(state)
        return action.detach().numpy()[0]

    def learn(self, episodes, episode_steps):
        max_steps = episode_steps
        for ep in tqdm(range(episodes)):
            state = self.env.reset()
            done = False
            i = 0
            score = 0
            while not done and i < max_steps:
                action = self.choose_action(state)
                new_state, reward, done, info = self.env.step(action)
                score += reward
                self.buffer.store_transition(state, action, reward, new_state, done)
                self._learn()
                i += 1
            print(f'episode: {ep}, score: {score}')

    def _learn(self):
        if len(self.buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done).to(self.device)

        _, log_probs = self.policy.sample_action(state)
        log_probs = log_probs.view(-1)

        # update value function
        value = self.value(state).view(-1)
        target_value = self.target_value(next_state).view(-1)  # target value of next_state
        target_value[done] = 0.0  # value is 0 at terminal states (definition of value functions)

        q1_old_policy = self.qf_1(state, action)
        q2_old_policy = self.qf_2(state, action)
        q_value = torch.min(q1_old_policy, q2_old_policy).view(-1)

        self.value.optimizer.zero_grad()
        value_loss = 0.5 * F.mse_loss(value, (q_value - log_probs))  # EQ (5)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # update both critics
        self.qf_1.optimizer.zero_grad()
        self.qf_2.optimizer.zero_grad()
        q_hat = reward * self.reward_scale + self.gamma * target_value
        q_hat = q_hat.unsqueeze(-1)
        q1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        q2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        q_loss = q1_loss + q2_loss
        q_loss.backward(retain_graph=True)
        self.qf_1.optimizer.step()
        self.qf_2.optimizer.step()

        # update actor
        self.policy.optimizer.zero_grad()
        actions, log_probs = self.policy.sample_action(state, reparam=True)
        q1_reparam_policy = self.qf_1(state, actions)
        q2_reparam_policy = self.qf_2(state, actions)
        q_value = torch.min(q1_reparam_policy, q2_reparam_policy).view(-1)
        policy_loss = torch.mean(log_probs - q_value)
        policy_loss.backward()
        self.policy.optimizer.step()

        self.update_value_target()

    def update_value_target(self, tau=None):
        if tau is None:
            tau = self.tau


        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict.keys():
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                (1-tau) * target_value_state_dict[name].clone()  # soft or hard update the network

        self.target_value.load_state_dict(value_state_dict)


class SAC_v3:

    def __init__(self, env, state_dim, action_dim,
                 batch_size=256, lr_policy=3e-4, lr_q=3e-4, init_weights: Optional[str] = None,
                 tau=0.005, reward_scale=1.0, buffer_max_size=1000000, hidden_dims = [256, 256],
                 target_update_interval=1, alpha=1
                 ):
        """
        Args:
            env: gym environment, used for training.
            state_dim: env.observation_space.shape[0]
            action_dim: env.action_space.shape[0]
            tau: target update factor
            alpha: temperature
        """

        self.env = env
        self.batch_size = batch_size
        self.lr_policy = lr_policy
        self.lr_q = lr_q
        self.tau = tau
        self.reward_scale = reward_scale
        self.buffer_max_size = buffer_max_size
        self.target_update_interval = target_update_interval

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.target_entropy = -torch.prod(self.env.action_space.shape).to(self.device)  # scalar value
        self.alpha = torch.zeros(1, requires_grad=True, device=self.device)  # learnable param, tensor?
        self.alpha_optim = optim.Adam([self.alpha], lr=self.lr_q)

        self.policy = PolicyNetwork(state_dim, action_dim, action_scaler=self.env.action_space.high,
                                    lr=lr_policy, device=self.device)
        self.qf_1 = QNetwork(state_dim, action_dim, lr=lr_q, name='qf_1', device=self.device)
        self.qf_2 = QNetwork(state_dim, action_dim, lr=lr_q, name='qf_2', device=self.device)

    def _learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, new_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done).to(self.device)

        '''
        THIS IS WRONG!!!
        need a target network
        '''
        # sample new actions
        actions, log_probs = self.policy.sample_action(state, reparam=False)
        log_probs = log_probs.view(-1)

        # calculate q_value with old policy
        q1_old_policy = self.qf_1(state, action)
        q2_old_policy = self.qf_2(state, action)

        # calculate q_value with new policy
        q1_new_policy = self.qf_1(state, actions)
        q2_new_policy = self.qf_2(state, actions)

        soft_q_val = torch.min(q1_new_policy, q2_new_policy)
        soft_q_val = soft_q_val.view(-1)

        soft_state_val = soft_q_val - self.alpha * log_probs
        # next q_value
        q_hat = reward * self.reward_scale + self.gamma * soft_state_val

        self.qf_1.optimizer.zero_grad()
        q1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        q2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        q_loss = q1_loss + q2_loss



class ValueNetwork(nn.Module):
    """
    This network judges the state.
    """

    def __init__(self, input_dim: int, hidden_dims: list = [256, 256], lr=3e-4,
                 name='value', device=None, **kwargs):
        super(ValueNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.name = name

        network = [nn.Linear(self.input_dim, self.hidden_dims[0]), nn.ReLU(inplace=True)]
        for idx, dim in enumerate(self.hidden_dims[:-1]):
            network.append(nn.Linear(dim, self.hidden_dims[idx+1]))
            network.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*network)
        self.out = nn.Linear(self.hidden_dims[-1], 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, **kwargs)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.to(self.device)

    def forward(self, state):
        x = self.network(state)
        return self.out(x)


class QNetwork(nn.Module):
    """
    This network approximates the Q-function. "Judges" the action taken.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256],
                 action_scaler: float = 1.0, name: str = 'actor', lr=3e-4, device=None, **kwargs):

        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.action_scaler = action_scaler
        self.name = name
        self.lr = lr

        network = [nn.Linear(self.state_dim + self.action_dim, self.hidden_dims[0]), nn.ReLU(inplace=True)]
        for idx, dim in enumerate(self.hidden_dims[:-1]):
            network.append(nn.Linear(dim, self.hidden_dims[idx+1]))
            network.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*network)
        self.out = nn.Linear(self.hidden_dims[-1], 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, **kwargs)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.network(x)
        return self.out(x)


# TODO: test with Gaussian Mixture Model policy ?
# https://pytorch.org/docs/stable/distributions.html#mixturesamefamily
class PolicyNetwork(nn.Module):
    """
    This is the policy network. Chooses an action with respect to the state.
    This implementation uses the gaussian policy.
    """

    def __init__(self, state_dim: int, action_dim: int, action_scaler: tuple, hidden_dims: list = [256, 256],
                 name: str = 'actor', noise=1e-6, lr=3e-4, device=None, **kwargs):
        super(PolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scaler = action_scaler
        self.hidden_dims = hidden_dims
        self.noise = noise
        self.name = name
        """
        Args:
            action_scaler: max value of the action space if it's not [-1, 1], so that we can scale the output value
        """

        network = [nn.Linear(self.state_dim, self.hidden_dims[0]), nn.ReLU(inplace=True)]
        for idx, dim in enumerate(self.hidden_dims[:-1]):
            network.append(nn.Linear(dim, self.hidden_dims[idx+1]))
            network.append(nn.ReLU(inplace=True))  # reduce memory

        self.network = nn.Sequential(*network)

        self.mu = nn.Linear(self.hidden_dims[-1], self.action_dim)
        self.sigma = nn.Linear(self.hidden_dims[-1], self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, **kwargs)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.to(self.device)

    def forward(self, state):
        prob = self.network(state)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.noise, max=1)  # add a small noise to avoid 0
        return mu, sigma

    def sample_action(self, state, reparam=False):
        mu, sigma = self.forward(state)
        probs = Normal(mu, sigma)

        if reparam:
            actions = probs.rsample()
        else:
            actions = probs.sample()

        # activate the chosen action and scale it to resemble the environment's scale
        action = torch.tanh(actions) * torch.tensor(self.action_scaler).to(self.device)
        log_probs = probs.log_prob(actions)  # log_prob, because it's more stable to optimize

        # Appendix C, https://arxiv.org/pdf/1812.05905v2.pdf
        log_probs -= torch.log(1 - action.pow(2) + self.noise)  # to avoid log(0)
        log_probs = log_probs.sum(1, keepdim=True)  # need a scalar quantity for the loss
        return action, log_probs


class ReplayBuffer:

    def __init__(self, max_size, state_dim, action_dim):
        self.mem_size = max_size
        self.mem_len = 0
        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.new_state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size, action_dim))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_len % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        self.mem_len += 1

    def __len__(self):
        return self.mem_len

    def sample(self, batch_size):
        max_mem = min(self.mem_len, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones
