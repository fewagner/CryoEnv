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

from ._rlutils import ReplayBuffer, Agent, OfflineTrajectoryBuffer

import torch
import torch.nn as nn


class QNetworkLSTM(nn.Module):
    # TODO adapt to LSTM

    def __init__(self, n_observations, n_actions, hidden_dims=[256, 256], device="cpu"):
        super(QNetworkLSTM, self).__init__()
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

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], 1)

        x1 = self.network_1(x)
        x2 = self.network_2(x)

        return x1, x2


class GaussianPolicyLSTM(nn.Module):
    # TODO adapt to LSTM

    def __init__(
            self,
            n_observations,
            n_actions,
            hidden_dims=[256, 256],
            device="cpu",
            noise=1e-6,
    ):
        super(GaussianPolicyLSTM, self).__init__()
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

    def sample(self, observations, greedy=False):
        mu, log_std = self.forward(observations)
        sigma = log_std.exp()

        probs = Normal(mu, sigma)
        sample = probs.rsample() if not greedy else mu
        actions = torch.tanh(sample)

        log_probs = probs.log_prob(sample)
        log_probs -= torch.log(1 - actions.pow(2) + self.noise)
        log_probs = log_probs.sum(dim=1, keepdim=True)

        return actions, log_probs


class OfflineSoftActorCritic(Agent):
    POLICIES = {"GaussianPolicyLSTM": GaussianPolicyLSTM}
    CRITICS = {"QNetworkLSTM": QNetworkLSTM}

    def __init__(
            self,
            buffer: OfflineTrajectoryBuffer,
            policy: str = "GaussianPolicyLSTM",
            critic: str = "QNetworkLSTM",
            lr=3e-4,
            weight_decay=1e-5,
            hidden_dims=[256, 256],
            batch_size=256,
            tau=0.005,  # update factor
            gamma=0.99,  # discount factor
            temperature=0.2,  # initial entropy coefficient
            target_update_interval=1,
            device="cpu",
            entropy_tuning=True,  # activate automatic entropy tuning
            grad_clipping=0.5,
    ):
        self.device = device
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = torch.FloatTensor([gamma]).to(self.device)
        self.alpha = torch.FloatTensor([temperature]).to(self.device)
        self.target_update_interval = target_update_interval
        self.entropy_tuning = entropy_tuning
        self.grad_clipping = grad_clipping

        policy_builder = self.POLICIES.get(policy, GaussianPolicyLSTM)
        critic_builder = self.CRITICS.get(critic, QNetworkLSTM)

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

        self.buffer = buffer
        self.buffer_size = len(self.buffer)

        self.target_entropy = -torch.prod(
            torch.tensor(self.env.action_space.shape).to(self.device)
        ).item()

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam(
            [self.log_alpha], lr=lr, weight_decay=weight_decay
        )
        self.gradient_steps = int(self.buffer_size / self.batch_size) + 1

    def learn(self, epochs, writer=None):
        self.policy.train()
        self.critic.train()
        self.target_critic.train()
        pbar = tqdm(range(epochs), leave=True)
        for epoch in pbar:

            for j in range(self.gradient_steps):
                update_target_value = True if self.buffer.buffer_total % self.target_update_interval == 0 else False
                self._learn_step(update_target_value=update_target_value, writer=writer)

                pbar.set_description(f"epoch: {epoch}/{epochs}, gradient steps: {j}/{self.total_num_steps}")

    def predict(self, state, greedy=True):
        self.policy.eval()

        state = torch.tensor(state).float().to(self.device)

        if len(state.shape) != 2:
            state = state.unsqueeze(0)

        action, log_probs = self.policy.sample(state, greedy=greedy)

        action = action.detach().cpu().numpy()
        log_probs = log_probs.detach().cpu().numpy()

        return action, log_probs

    def reset(self):
        # TODO reset internal state of policy, and critic
        pass

    def _learn_step(self, update_target_value, writer=None):
        # TODO adapt to LSTM
        state, action, reward, next_state, terminal = self.buffer.sample_buffer(self.batch_size)
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

        if self.entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

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

    def _choose_action(self, state):
        state_tensor = torch.tensor(np.array([state])).float().to(self.device)
        action, _ = self.policy.sample(state_tensor)
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
            policy: str = "GaussianPolicyLSTM",
            critic: str = "QNetworkLSTM",
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
