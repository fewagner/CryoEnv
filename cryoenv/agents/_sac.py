from typing import List, Union, Tuple, Type

import gym
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from cryoenv.base import Agent, Policy, ValueFunction
from cryoenv.buffers import ReplayBuffer
from cryoenv.agents.sac import Actor, ValueNetwork, Critic


class SAC(Agent):

    def __init__(self,
                 env: gym.Env, policy: Type[Actor],
                 value_function: Type[ValueNetwork],
                 soft_q: Type[Critic],
                 buffer = None,
                 batch_size: int = 256,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_value_func: float = 3e-4,
                 hidden_dims: list = [256, 256],
                 tau: float = 0.005,  # soft or hard update of the target value network
                 reward_scale: float = 2.0,
                 buffer_size: int = 1_000_000,
                 gamma: float = 0.99,
                 target_update_interval: int = 1,
                 device: str = 'cpu') -> None:
        super(SAC, self).__init__(env, policy, value_function)
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_value_func = lr_value_func
        self.tau = tau
        self.reward_scale = reward_scale
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval

        self.policy = policy(lr=self.lr_actor)
        self.value_function = value_function(lr=self.lr_value_func)
        self.target_value_function = value_function(lr=self.lr_value_func)
        self.critic_1 = soft_q(self.lr_critic)
        self.critic_2 = soft_q(self.lr_critic)
        if buffer is None:
            self.buffer = ReplayBuffer(buffer_size, (self.nmbr_observations,), self.nmbr_actions)
        else:
            self.buffer = buffer

        self.device = device
        self.define_spaces()

    def learn(self, episodes, episode_steps) -> None:
        self.train()
        pbar = tqdm(range(episodes))
        for episode in pbar:
            observation = self.env.reset()
            done = False
            i = 0
            score = 0
            while not done and i < episode_steps:
                action = self._choose_action(observation)
                new_observation, reward, done, info = self.env.step(action)
                score += reward
                self.buffer.store_transition(observation, action, reward, new_observation, done)
                self._learn_step(i)
                i += 1
                observation = new_observation
            pbar.set_description(f"episode: {episode}, score: {score}")
        pbar.close()

    def define_spaces(self):
        self.policy.define_spaces(self.env.action_space, self.env.observation_space, None)
        self.value_function.define_spaces(self.env.action_space, self.env.observation_space)
        self.target_value_function.define_spaces(self.env.action_space, self.env.observation_space)
        self.critic_1.define_spaces(self.env.action_space, self.env.observation_space)
        self.critic_2.define_spaces(self.env.action_space, self.env.observation_space)

    def train(self) -> None:
        if self._is_training():
            return
        else:
            self.policy.train()
            self.value_function.train()
            self.critic_1.train()
            self.critic_2.train()

    def predict(self, observation) -> torch.Tensor:
        if self.policy.training:  # set the model to evaluation mode
            self.policy.eval()
        return self.policy(observation)

    def _is_training(self) -> bool:
        if all([self.policy.training,
                self.value_function.training,
                self.critic_1.training,
                self.critic_2.training]):
            return True
        else:
            return False

    def _choose_action(self, observation: Union[list, np.ndarray]) -> torch.Tensor:
        observation = torch.tensor(np.array([observation])).float().to(self.device)
        action, _ = self.policy.predict(observation)
        return action

    def _learn_step(self, gradient_step: int = 1) -> None:
        # if the buffer is not filled with enough data,
        # do nothing
        if len(self.buffer) < self.batch_size:
            return

        observation, action, reward, \
            next_observation, done = self.buffer.sample_buffer(self.batch_size)
        observation = torch.tensor(observation, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        next_observation = torch.tensor(next_observation, dtype=torch.float).to(self.device)
        done = torch.tensor(done).to(self.device)

        value = self.value_function(observation).view(-1)
        target_value = self.target_value_function(next_observation).view(-1)
        target_value[done] = 0.0  # value is 0 at terminal observations

        next_actions, log_probs = self.policy.predict(observation)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1(next_actions, observation)
        q2_new_policy = self.critic_2(next_actions, observation)
        soft_q = torch.min(q1_new_policy, q2_new_policy).view(-1)

        self.value_function.zero_grad()
        value_loss = 0.5 * F.mse_loss(value, (soft_q - log_probs))  # EQ (5)
        # don't disconnect from the graph, as we do further optimizations
        value_loss.backward(retain_graph=True)
        self.value_function.update()


        # update Q approximator (critic)
        self.critic_1.zero_grad()
        self.critic_2.zero_grad()
        q_hat = reward * self.reward_scale + self.gamma * target_value

        q1_old_policy = self.critic_1(action, observation).view(-1)
        q2_old_policy = self.critic_2(action, observation).view(-1)

        q1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        q2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        q_loss = q1_loss + q2_loss

        q_loss.backward(retain_graph=True)
        self.critic_1.update()
        self.critic_2.update()

        # update policy
        # use reparam trick to make the log_probs differentiable
        actions, log_probs = self.policy.predict(observation, reparam=True)
        log_probs = log_probs.view(-1)

        q1_reparam_policy = self.critic_1(actions, observation)
        q2_reparam_policy = self.critic_2(actions, observation)
        soft_q = torch.min(q1_reparam_policy, q2_reparam_policy).view(-1)

        self.policy.zero_grad()
        policy_loss = torch.mean(log_probs - soft_q)
        policy_loss.backward()
        self.policy.update()

        if gradient_step % self.target_update_interval == 0:
            self._update_target_value()

    def _update_target_value(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value_function.named_paramaters()
        value_params = self.value_function.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict.keys():
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                (1-tau) * target_value_state_dict[name].clone()

        self.target_value_function.load_state_dict(value_state_dict)
