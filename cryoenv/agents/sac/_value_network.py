from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from cryoenv.base import ValueFunction


class ValueNetwork(ValueFunction, nn.Module):

    def __init__(self,
                 lr: float = 3e-4,
                 hidden_dims: List[int] = [256, 256],
                 inplace: bool = False,
                 device: str = 'cpu',
                 ) -> None:
        super(ValueFunction, self).__init__()
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.device = device
        self.inplace = inplace

    def forward(self, observation) -> torch.Tensor:
        x = self.network(observation)
        return self.out(x)

    def create_network(self, **kwargs):
        network = [nn.Linear(self.nmbr_observations, self.hidden_dims[0]),
                   nn.ReLU(inplace=self.inplace)]
        for idx, dim in enumerate(self.hidden_dims[:-1], 1):
            network.append(nn.Linear(dim, self.hidden_dims[idx]))
            network.append(nn.ReLU(inplace=self.inplace))

        self.network = nn.Sequential(*network)
        self.out = nn.Linear(self.hidden_dims[-1], 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, **kwargs)

        self.to(self.device)

    def _setup(self, **kwargs):
        self.create_network(**kwargs)

    def predict(self, observation):
        return self.forward(observation)

    def update(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
