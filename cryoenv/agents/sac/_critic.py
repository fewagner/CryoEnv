from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import gym


class Critic(nn.Module):

    def __init__(self,
                 lr: float = 3e-4,
                 hidden_dims: List[int] = [256, 256],
                 inplace: bool = False,
                 device: str = 'cpu',
                 ):
        super(Critic, self).__init__()
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.device = device
        self.inplace = inplace

    def define_spaces(self, action_space, observation_space) -> None:
        assert type(action_space) == gym.spaces.box.Box, 'action_space must be  of type gym.spaces.box.Box!'
        assert type(observation_space) == gym.spaces.box.Box, 'observation_space must be  of type gym.spaces.box.Box!'
        assert all(action_space.low == -1), 'action_space.low must all be -1!'
        assert all(action_space.high == 1), 'action_space.high must all be 1!'
        assert all(observation_space.low == -1), 'observation_space.low must all be -1!'
        assert all(observation_space.high == 1), 'observation_space.high must all be 1!'

        self.action_space = action_space
        self.observation_space = observation_space
        self.nmbr_actions = action_space.shape[0]
        self.nmbr_observations = observation_space.shape[0]
        self._setup()

    def create_network(self, **kwargs) -> None:
        network = [nn.Linear(self.nmbr_actions + self.nmbr_observations, self.hidden_dims[0]),
                   nn.ReLU(inplace=self.inplace)]
        for idx, dim in enumerate(self.hidden_dims[:-1], 1):
            network.append(nn.Linear(dim, self.hidden_dims[idx]))
            network.append(nn.ReLU(inplace=self.inplace))

        self.network = nn.Sequential(*network)
        self.out = nn.Linear(self.hidden_dims[-1], 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, **kwargs)
        self.to(self.device)

    def _setup(self) -> None:
        self.create_network()

    def forward(self, action, observation) -> torch.Tensor:
        x = torch.cat([action, observation], dim=1)
        x = self.network(x)
        return self.out(x)

    def update(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
