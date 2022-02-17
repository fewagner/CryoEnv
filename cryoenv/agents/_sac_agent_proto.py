import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

from cryoenv.agents import Agent


class SAC:

    def __init__(self, alpha=3e-4, beta=3e-4, input_dims=[8], env=None, gamma=0.99, tau=0.005, n_actions=2,
                 max_size=1000000, fc1_dims=256, fc2_dims=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.env = env

        self.actor = Actor(alpha, input_dims, n_actions=self.n_actions, name='actor',
                           max_action=self.env.action_space.high)
        self.critic_1 = Critic(beta, input_dims, n_actions=self.n_actions, name='critic_1')
        self.critic_2 = Critic(beta, input_dims, n_actions=self.n_actions, name='critic_2')

        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')  # we will softcopy the value network

        self.scale = reward_scale
        self.update_network_parameters()

    def choose_action(self, observation):
        state = torch.tensor(np.array([observation])).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparam=True)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
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

    def save_models(self, steps=100):
        print('...saving models...')
        self.actor.save_checkpoint(steps)
        self.value.save_checkpoint(steps)
        self.target_value.save_checkpoint(steps)
        self.critic_1.save_checkpoint(steps)
        self.critic_2.save_checkpoint(steps)

    def load_models(self, steps=100):
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0  # value is 0 at terminal states (definition of value function)

        actions, log_probs = self.actor.sample_normal(state, reparam=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)  # critic value under the new policy
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)  # to eliminate the overestimation bias
        critic_value = critic_value.view(-1)

        # value network loss
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)  # EQ (5)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparam=True)
        log_probs = log_probs.view(-1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        #print(critic_loss.item())
        critic_loss.backward(retain_graph=True)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        #print(f'Actor loss: {actor_loss.item():.1f}')
        self.actor.optimizer.step()

        self.update_network_parameters()


# Networks
class Critic(nn.Module):

    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/sac', name='critic'):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        action_value = F.relu(self.fc2(action_value))
        q = self.q(action_value)
        return q

    def save_checkpoint(self, steps = 100):
        torch.save(self.state_dict(), self.chkpt_file + str(steps))

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class ValueNetwork(nn.Module):

    def __init__(self, beta, input_dims, name='value', fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))
        return self.v(state_value)

    def save_checkpoint(self, steps = 100):
        torch.save(self.state_dict(), self.chkpt_file + str(steps))

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class Actor(nn.Module):

    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256,
                 n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(Actor, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name+'_sac')
        self.max_action = max_action  # scale up the action from [-1;1] to the environment scale
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = torch.sigmoid(sigma)
        #sigma = torch.clamp(sigma, min=0.000001, max=1)
        return mu, sigma

    def sample_normal(self, state, reparam=True):
        mu, sigma = self.forward(state.float())
        probs = Normal(mu, sigma)

        if reparam:
            actions = probs.rsample()
        else:
            actions = probs.sample()

        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)  # here, we activate the action to be in [-1, 1] and scale it
        log_probs = probs.log_prob(actions)  # log_probs, because it's more stable to optimize

        # comes from the appendix of the original SAC paper, else it could be log(0)
        # appendix (21) -> Enforcing Action Bounds
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)

        log_probs = log_probs.sum(1, keepdim=True)  # we need a scalar quantity for the loss
        return action, log_probs

    def save_checkpoint(self, steps = 100):
        torch.save(self.state_dict(), self.chkpt_file + str(steps))

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class ReplayBuffer:

    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones
