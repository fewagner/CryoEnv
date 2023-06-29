import os
import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.optim as optim
from tqdm.auto import tqdm

from ._rlutils import ReplayBuffer, Agent

import torch
import torch.nn as nn


class CryoWorldModel(nn.Module):
    # tuned to 7 obs: (ph, rms, ib, dac, tpa, ib_m, dac_m), 2 act: (dac, ib)
    # TODO adapt to no memory states and TPA

    def __init__(self, memory_factor=0.8, curiosity_factor=0.1, hidden_dims=[256, 256], lr=3e-4, weight_decay=1e-5,
                 device="cpu"):
        super(CryoWorldModel, self).__init__()
        self.device = device

        response_network = []
        input_channels = 5  # dac, ib, tpa, dac_m, ib_m
        for n_channels in hidden_dims:
            response_network.append(nn.Linear(input_channels, n_channels))
            response_network.append(nn.ReLU())

            input_channels = n_channels

        response_network.append(nn.Linear(input_channels, 2))  # ph_, rms_
        self.response_network = nn.Sequential(*response_network)

        curiosity_network = []
        input_channels = 4  # dac, ib, dac_m, ib_m
        for n_channels in hidden_dims:
            curiosity_network.append(nn.Linear(input_channels, n_channels))
            curiosity_network.append(nn.ReLU())

            input_channels = n_channels

        curiosity_network.append(nn.Linear(input_channels, 1))  # ph_, rms_
        self.curiosity_network = nn.Sequential(*curiosity_network)

        self.to(self.device)

        self.response_optim = optim.Adam(self.response_network.parameters(), lr=lr, weight_decay=weight_decay)
        self.curiosity_optim = optim.Adam(self.curiosity_network.parameters(), lr=lr, weight_decay=weight_decay)
        self.memory_factor = memory_factor
        self.curiosity_factor = curiosity_factor

    def train_batch(self, state, action, next_state, reward):

        # train response net
        x = next_state[:, 2:]  # ib_, dac_, tpa_, ib_m_, dac_m_
        pred = self.response_network(x)

        y = next_state[:, :2]  # ph_, rms_
        response_loss = F.mse_loss(pred, y)
        self.response_optim.zero_grad()
        response_loss.backward(retain_graph=True)
        self.response_optim.step()

        # train curiosity net
        x = next_state[:, [2, 3, 5, 6]]  # ib_, dac_, ib_m_, dac_m_
        pred = torch.sigmoid(self.curiosity_network(x))
        y = torch.zeros((len(x)))

        x_ = torch.rand(x.shape)  # ib_, dac_, ib_m_, dac_m_
        pred_ = torch.sigmoid(self.curiosity_network(x))
        y_ = torch.ones((len(x)))

        curiosity_loss = F.mse_loss(torch.cat((pred, pred_), 0), torch.cat((y, y_), 0))
        self.curiosity_optim.zero_grad()
        curiosity_loss.backward(retain_graph=True)
        self.curiosity_optim.step()

        return response_loss.item(), curiosity_loss.item()

    def get_next_state(self, state, action):
        # predicts next state
        # ph, rms, ib, dac, tpa, ib_m, dac_m = np.array(state).T

        next_state = np.zeros(state.shape, dtype=float)
        next_state[:, 2:4] = action  # dac, ib
        next_state[:, 6] = state[:, 6] * self.memory_factor - (1 - self.memory_factor) * action[:, 0]  # dac_m
        next_state[:, 5] = next_state[:, 5] * self.memory_factor - (1 - self.memory_factor) * action[:, 1]  # ib_m
        next_state[:, 4] = np.random.uniform(-1, 1, size=len(state))  # tpa
        x = torch.from_numpy(state[:, 2:]).float().to(self.device)
        pred = self.response_network(x).detach().cpu().numpy()
        next_state[:, :2] = pred  # ph, rms

        terminal = np.zeros(len(state), dtype=bool)
        return next_state, terminal

    def get_reward(self, state, action, next_state):
        # predicts next reward
        ph = next_state[:, 0]
        rms = next_state[:, 1]
        tpa = next_state[:, 4]
        x = torch.from_numpy(state[:, [2, 3, 5, 6]]).float().to(self.device)  # ib_, dac_, ib_m_, dac_m_
        pred = self.curiosity_network(x).detach().cpu().numpy().flatten()
        reward = - rms * tpa / ph
        reward += 1 / (1 + np.exp(pred)) * self.curiosity_factor
        return reward


class QNetwork(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_dims=[256, 256], device="cpu", n_grid=2000):
        super(QNetwork, self).__init__()
        self.device = device
        network_1 = []
        network_2 = []

        input_channels = n_observations + n_actions
        for n_channels in hidden_dims:
            network_1.append(nn.Linear(input_channels, n_channels))
            network_1.append(nn.InstanceNorm1d(n_channels))
            network_1.append(nn.ReLU())

            network_2.append(nn.Linear(input_channels, n_channels))
            network_2.append(nn.InstanceNorm1d(n_channels))
            network_2.append(nn.ReLU())

            input_channels = n_channels

        network_1.append(nn.Linear(input_channels, 1))
        network_2.append(nn.Linear(input_channels, 1))

        self.network_1 = nn.Sequential(*network_1)
        self.network_2 = nn.Sequential(*network_2)

        self.to(self.device)
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.n_grid = n_grid

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], 1)

        x1 = self.network_1(x)
        x2 = self.network_2(x)

        return x1, x2

    def greedy(self, observations):
        observations = torch.from_numpy(np.array(observations)).float().to(self.device).reshape(1,-1)

        def f(a):
            x1, x2 = self.forward(observations, a)
            return torch.minimum(x1, x2)

        class Actions(nn.Module):

            def __init__(self, n_actions):
                super(Actions, self).__init__()
                self.actions = torch.nn.Parameter(torch.zeros((1, n_actions)), requires_grad=True)

            def forward(self):
                x = self.actions
                return torch.tanh(x)

        actions = Actions(self.n_actions).to(self.device)
        optim = torch.optim.Adam(actions.parameters(), lr=1e-2)

        for i in range(500):
            optim.zero_grad()
            loss = -f(actions())
            loss.backward()
            optim.step()

        return actions().detach().cpu().numpy().flatten()


class GaussianPolicy(nn.Module):

    def __init__(
            self,
            n_observations,
            n_actions,
            hidden_dims=[256, 256],
            device="cpu",
            noise=1e-6,
    ):
        super(GaussianPolicy, self).__init__()
        self.device = device
        self.noise = noise

        layers = []
        input_channels = n_observations
        for n_channels in hidden_dims:
            layers.append(nn.Linear(input_channels, n_channels))
            layers.append(nn.InstanceNorm1d(n_channels))
            layers.append(nn.ReLU())
            input_channels = n_channels

        self.network = nn.Sequential(*layers)
        self.mu = nn.Linear(input_channels, n_actions)
        self.log_std = nn.Linear(input_channels, n_actions)

        self.to(self.device)

    def forward(self, observations):
        x = self.network(observations)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

    @staticmethod
    def gauss(x, mu, sigma):
        return torch.exp( - ((x - mu) / sigma) ** 2 / 2) / sigma / np.sqrt(2 * np.pi)

    @staticmethod
    def tanhgauss(y, mu, sigma):
        return torch.exp(- ((torch.arctanh(y) - mu) / sigma) ** 2 / 2) / sigma / np.sqrt(2 * np.pi) / torch.abs(1 - y ** 2)

    def sample(self, observations, greedy=False):
        mu, log_std = self.forward(observations)
        sigma = log_std.exp()

        probs = Normal(mu, sigma)
        sample = probs.rsample() if not greedy else mu
        actions = torch.tanh(sample)

        log_probs = probs.log_prob(sample)
        log_probs -= torch.log(1 - actions.pow(2) + self.noise)
        # log_probs = torch.log(self.tanhgauss(actions, mu, sigma))  # equivalent to the above lines
        log_probs = log_probs.sum(dim=1, keepdim=True)

        return actions, log_probs


class SoftActorCritic(Agent):
    POLICIES = {"GaussianPolicy": GaussianPolicy}
    CRITICS = {"QNetwork": QNetwork}

    def __init__(
            self,
            env: gym.Env,
            policy: str = "GaussianPolicy",
            critic: str = "QNetwork",
            world_model: nn.Module = None,
            lr=3e-4,
            weight_decay=1e-5,
            hidden_dims=[256, 256],
            buffer=None,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,  # update factor
            gamma=0.99,  # discount factor
            temperature=0.2,  # initial entropy coefficient alpha
            target_update_interval=1,
            device="cpu",
            entropy_tuning=True,  # activate automatic entropy tuning
            gradient_steps=1,
            model_steps=1,
            learning_starts=1,
            grad_clipping=0.5,
            target_entropy=None,  # float or None
            target_entropy_reduction=.9999,  # after 10000 gradient steps, target entropy is ~ 0.35 * target_entropy
            target_entropy_std=0.088,
    ):
        self.device = device
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = torch.FloatTensor([gamma]).to(self.device)
        self.alpha = torch.FloatTensor([temperature]).to(self.device)
        self.target_update_interval = target_update_interval
        self.entropy_tuning = entropy_tuning
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        self.grad_clipping = grad_clipping

        self.env = env

        policy_builder = self.POLICIES.get(policy, GaussianPolicy)
        critic_builder = self.CRITICS.get(critic, QNetwork)

        self.policy = policy_builder(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_dims=hidden_dims,
            device=self.device,
        )
        self.policy_optim = optim.Adam(
            self.policy.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.critic = critic_builder(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_dims=hidden_dims,
            device=self.device,
        )
        self.critic_optim = optim.Adam(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.target_critic = deepcopy(self.critic)

        if buffer is None:
            self.buffer = ReplayBuffer(
                int(buffer_size),
                (self.env.observation_space.shape[0],),
                self.env.action_space.shape[0],
            )
        else:
            self.buffer = buffer

        self.action_dim = torch.prod(
                torch.tensor(self.env.action_space.shape).to(self.device)
            ).item()

        if target_entropy is None:
            self.target_entropy = self.entropy_gaussian(std=target_entropy_std, d=self.action_dim)
            # for std=0.088 this gives the same as the original SAC implementation (-dim action_space)
        else:
            self.target_entropy = target_entropy

        self.log_alpha = torch.tensor(torch.log(self.alpha), requires_grad=True, device=self.device)
        self.alpha_optim = optim.SGD(
            [self.log_alpha], lr=lr, weight_decay=weight_decay
        )
        self.total_num_steps = 0
        self.total_gradient_steps = 0

        self.world_model = world_model
        self.model_steps = model_steps

        self.target_entropy_reduction = target_entropy_reduction
        self.target_entropy_std = target_entropy_std

    @staticmethod
    def entropy_gaussian(std, d):
        return d / 2. * np.log(2. * np.pi * np.e * std ** 2)

    def learn(self, episodes: int = 1, episode_steps: int = 100, writer=None, two_pbars=True, tracker=None):
        self.policy.train()
        self.critic.train()
        self.target_critic.train()
        self._fill_buffer()
        pbar = tqdm(range(episodes), leave=True)
        for episode in pbar:

            if tracker is not None:
                tracker.new_episode()

            state, info = self.env.reset()  # is_training=True
            return_ = 0
            iterator = range(episode_steps)
            if two_pbars:
                iterator = tqdm(iterator, leave=False)
            for i in iterator:
                action = self._choose_action(state, greedy=False).cpu().detach().numpy()[0]
                next_state, reward, terminated, truncated, info = self.env.step(action)
                return_ += reward
                if tracker is not None:
                    tracker.add(reward)
                self.buffer.store_transition(state, action, reward, next_state, terminated)
                if self.buffer.buffer_total > self.learning_starts:
                    for j in range(self.gradient_steps):
                        update_target_value = True if self.buffer.buffer_total % self.target_update_interval == 0 else False
                        self._learn_step(update_target_value=update_target_value, writer=writer)
                state = next_state
                self.total_num_steps += 1

                if terminated or truncated:
                    if two_pbars:
                        iterator.disp(close=True)
                    break
            pbar.set_description(f"total steps: {self.total_num_steps}, episode: {episode}, return: {return_:.4f}")

    def predict(self, state, greedy=True):
        self.policy.eval()

        state = torch.tensor(state).float().to(self.device)

        if len(state.shape) != 2:
            state = state.unsqueeze(0)

        action, log_probs = self.policy.sample(state, greedy=greedy)

        action = action.detach().cpu().numpy()
        log_probs = log_probs.detach().cpu().numpy()

        return action, log_probs

    def _fill_buffer(self):
        state, info = self.env.reset()
        while len(self.buffer) < self.batch_size:
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.buffer.store_transition(state, action, reward, next_state, terminated)
            state = next_state

    def _learn_model_step(self):  # not properly tested
        state = np.random.uniform(-1, 1, size=(self.batch_size, self.env.observation_space.shape[0]))
        action = np.random.uniform(-1, 1, size=(self.batch_size, self.env.action_space.shape[0]))
        next_state, terminal = self.world_model.get_next_state(state, action)
        reward = self.world_model.get_reward(state, action, next_state)

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device).view(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        terminal = torch.tensor(terminal, dtype=torch.bool).to(self.device).view(-1, 1)

        # calc next q values for TD update
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_state)
            next_critic1_target, next_critic2_target = self.target_critic(next_state, next_actions)
            next_min_critic_target = torch.min(next_critic1_target, next_critic2_target) - self.alpha * next_log_probs
            next_q_value = reward + torch.logical_not(terminal) * self.gamma * next_min_critic_target

        # train critic/values with mse loss
        critic_1, critic_2 = self.critic(state, action)
        critic_1_loss = F.mse_loss(critic_1, next_q_value)
        critic_2_loss = F.mse_loss(critic_2, next_q_value)
        critic_loss = critic_1_loss + critic_2_loss
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clipping)
        self.critic_optim.step()

        # train actor/policy with TD error
        actions, log_probs = self.policy.sample(state)
        next_critic_1, next_critic_2 = self.critic(state, actions)
        next_min_q = torch.min(next_critic_1, next_critic_2)
        policy_loss = ((self.alpha * log_probs) - next_min_q).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), self.grad_clipping)
        self.policy_optim.step()

    def _learn_step(self, update_target_value, writer=None):
        state, action, reward, next_state, terminal = self.buffer.sample_buffer(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device).view(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        terminal = torch.tensor(terminal, dtype=torch.bool).to(self.device).view(-1, 1)

        # calc next q values for TD update
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_state, greedy=False)  # could try greedy here!
            next_critic1_target, next_critic2_target = self.target_critic(next_state, next_actions)
            next_min_critic_target = torch.min(next_critic1_target, next_critic2_target) - self.alpha * next_log_probs
            next_q_value = reward + torch.logical_not(terminal) * self.gamma * next_min_critic_target

        # train critic/values with mse loss
        critic_1, critic_2 = self.critic(state, action)
        critic_1_loss = F.mse_loss(critic_1, next_q_value)
        critic_2_loss = F.mse_loss(critic_2, next_q_value)
        critic_loss = critic_1_loss + critic_2_loss
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clipping)
        self.critic_optim.step()

        # train actor/policy with TD error
        actions, log_probs = self.policy.sample(state, greedy=False)  # here is off policy
        next_critic_1, next_critic_2 = self.critic(state, actions)
        next_min_q = torch.min(next_critic_1, next_critic_2)
        policy_loss = ((self.alpha * log_probs) - next_min_q).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), self.grad_clipping)
        self.policy_optim.step()

        if self.entropy_tuning:
            alpha_loss = - (self.log_alpha * (log_probs + self.target_entropy).detach()).mean()  # - log_probs is entropy
            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

            self.target_entropy_std *= self.target_entropy_reduction
            self.target_entropy = self.entropy_gaussian(self.target_entropy_std, self.action_dim)

        if update_target_value:
            self._update_target_value()

        self.total_gradient_steps += 1

        if writer is not None:
            writer.add_scalar("loss/policy", policy_loss.item(), self.total_gradient_steps)
            writer.add_scalar(
                "loss/critic_1", critic_1_loss.item(), self.total_gradient_steps
            )
            writer.add_scalar(
                "loss/critic_2", critic_2_loss.item(), self.total_gradient_steps
            )
            if self.entropy_tuning:
                writer.add_scalar(
                    "loss/entropy", alpha_loss.item(), self.total_gradient_steps
                )
                writer.add_scalar(
                    "alpha",
                    self.alpha.clone().detach().cpu().numpy(),
                    self.total_gradient_steps,
                )

        if self.world_model is not None:

            for j in range(self.model_steps):
                model_loss, curiosity_loss = self.world_model.train_batch(state, action, next_state, reward)

                if writer is not None:
                    writer.add_scalar(
                        "loss/response", model_loss, self.total_gradient_steps
                    )
                    writer.add_scalar(
                        "loss/curiosity", curiosity_loss, self.total_gradient_steps
                    )

            for j in range(self.model_steps):
                self._learn_model_step()

    def _choose_action(self, state, greedy=False):
        state_tensor = torch.tensor(np.array([state])).float().to(self.device)
        action, _ = self.policy.sample(state_tensor, greedy=greedy)
        return action

    def _update_target_value(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_critic_param, critic_param in zip(
                self.target_critic.parameters(), self.critic.parameters()
        ):
            target_critic_param.data.copy_(
                target_critic_param.data * (1.0 - tau) + critic_param.data * tau
            )

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), os.path.join(path, "policy.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
        torch.save(
            self.target_critic.state_dict(), os.path.join(path, "target_critic.pt")
        )

    @classmethod
    def load(
            cls,
            env,
            path,
            policy: str = "GaussianPolicy",
            critic: str = "QNetwork",
            device="cpu",
            load_critic=False,
            **kwargs,
    ):
        """
        Loads the policy weights and optionally the critic for inference.
        """
        sac = cls(env, policy, critic, **kwargs)
        sac.policy.load_state_dict(
            torch.load(os.path.join(path, "policy.pt"), map_location=device)
        )
        if load_critic:
            sac.critic.load_state_dict(
                torch.load(os.path.join(path, "critic.pt"), map_location=device)
            )
            sac.target_critic.load_state_dict(
                torch.load(os.path.join(path, "target_critic.pt"), map_location=device)
            )
        return sac

    def train(self):
        self.policy.train()
        self.critic.train()

    def eval(self):
        self.policy.eval()
        self.critic.eval()
