import os
import time
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from cryoenv.buffers import ReplayBuffer

from .sac import GaussianPolicy, QNetwork


class SAC:
    POLICIES = {"GaussianPolicy": GaussianPolicy}
    CRITICS = {"QNetwork": QNetwork}

    def __init__(
        self,
        env: gym.Env,
        policy: str = "GaussianPolicy",
        critic: str = "QNetwork",
        lr=3e-4,
        weight_decay=1e-5,
        hidden_dims=[256, 256],
        buffer=None,
        buffer_init_steps=1000,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,  # update factor
        gamma=0.99,  # discount factor
        temperature=0.2,  # initial entropy coefficient
        target_update_interval=1,
        device="cpu",
        entropy_tuning=True,  # activate automatic entropy tuning
    ):
        self.device = device
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = torch.FloatTensor([gamma]).to(self.device)
        self.alpha = torch.FloatTensor([temperature]).to(self.device)
        self.target_update_interval = target_update_interval
        self.entropy_tuning = entropy_tuning

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
                buffer_size,
                (self.env.observation_space.shape[0],),
                self.env.action_space.shape[0],
            )
        else:
            self.buffer = buffer

        self.target_entropy = -torch.prod(
            torch.tensor(self.env.action_space.shape).to(self.device)
        ).item()

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam(
            [self.log_alpha], lr=lr, weight_decay=weight_decay
        )
        self.total_num_steps = 0

    def learn(self, episodes: int = 1, episode_steps: int = 100, writer=None):
        self.policy.train()
        self.critic.train()
        self.target_critic.train()
        self._fill_buffer()
        pbar = tqdm(range(episodes), leave=False)
        for episode in pbar:
            state = self.env.reset(is_training=True)
            score = 0
            for i in tqdm(range(episode_steps), leave=False):
                action = self._choose_action(state).cpu().detach().numpy()[0]
                next_state, reward, done, info = self.env.step(action)
                score += reward
                self.buffer.store_transition(state, action, reward, next_state, done)
                self._learn_step(episode, gradient_step=i, writer=writer)
                state = next_state
                self.total_num_steps += 1
                pbar.set_description(f"episode: {episode}, score: {score:.4f}")

            if episode % 10 == 0:
                obs = self.env.reset()
                for i in range(100):
                    _action = self.predict(obs)
                    obs, _, _, _ = self.env.step(_action.detach().cpu().numpy()[0])
                _rewards, _, _, _, _, _, _, _ = self.env.get_trajectory()
                self.policy.train()
                if writer is not None:
                    writer.add_scalar("validation/reward", np.mean(_rewards), episode)

    def predict(self, state):
        self.policy.eval()

        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().to(self.device)

        if len(state.shape) != 2:
            state = state.unsqueeze(0)

        action, _ = self.policy(state)
        return torch.tanh(action)

    def _fill_buffer(self):
        state = self.env.reset()
        while len(self.buffer) < self.batch_size:
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            self.buffer.store_transition(state, action, reward, next_state, done)
            state = next_state

    def _learn_step(self, episode, gradient_step, writer=None):
        state, action, reward, next_state, _ = self.buffer.sample_buffer(
            self.batch_size
        )
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device).view(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_state)
            next_critic1_target, next_critic2_target = self.target_critic(
                next_state, next_actions
            )
            next_min_critic_target = (
                torch.min(next_critic1_target, next_critic2_target)
                - self.alpha * next_log_probs
            )
            next_q_value = reward + self.gamma * next_min_critic_target

        critic_1, critic_2 = self.critic(state, action)
        critic_1_loss = F.mse_loss(critic_1, next_q_value)
        critic_2_loss = F.mse_loss(critic_2, next_q_value)
        critic_loss = critic_1_loss + critic_2_loss
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        actions, log_probs = self.policy.sample(state)
        next_critic_1, next_critic_2 = self.critic(state, actions)
        next_min_q = torch.min(next_critic_1, next_critic_2)
        policy_loss = ((self.alpha * log_probs) - next_min_q).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 0.5)
        self.policy_optim.step()

        if self.entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        if gradient_step % self.target_update_interval == 0:
            self._update_target_value()

        if writer is not None:
            writer.add_scalar("loss/policy", policy_loss.item(), self.total_num_steps)
            writer.add_scalar(
                "loss/critic_1", critic_1_loss.item(), self.total_num_steps
            )
            writer.add_scalar(
                "loss/critic_2", critic_2_loss.item(), self.total_num_steps
            )
            if self.entropy_tuning:
                writer.add_scalar(
                    "loss/entropy", alpha_loss.item(), self.total_num_steps
                )
                writer.add_scalar(
                    "alpha",
                    self.alpha.clone().detach().cpu().numpy(),
                    self.total_num_steps,
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
        policy: str = "GaussianPolicy",
        critic: str = "QNetwork",
        device="cpu",
    ):
        """
        Loads only the policy weights for inference.
        """
        sac = cls(env, policy, critic)
        sac.policy.load_state_dict(
            torch.load(os.path.join(path, "policy.pt"), map_location=device)
        )
        return sac
