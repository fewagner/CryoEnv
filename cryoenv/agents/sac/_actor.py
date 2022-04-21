from typing import List, Union, Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from cryoenv.base import Policy


class Actor(Policy, nn.Module):

    def __init__(self,
                 lr: float = 3e-4,
                 hidden_dims: List[int] = [256, 256],
                 noise: float = 1e-6,
                 scaler: List[Union[float, int]] = [-1, 1],
                 inplace: bool = False,
                 device: str = 'cpu') -> None:
        super(Policy, self).__init__()
        self.lr = lr
        self.hidden_dims = hidden_dims
        self.noise = noise
        self.device = device
        self.scaler = scaler
        self.inplace = inplace

    def _setup(self, **kwargs):
        self.create_network(**kwargs)

    def create_network(self, **kwargs):
        network = [nn.Linear(self.nmbr_observations, self.hidden_dims[0]),
                   nn.ReLU(inplace=self.inplace)]
        for idx, dim in enumerate(self.hidden_dims[:-1], 1):
            network.append(nn.Linear(dim, self.hidden_dims[idx]))
            network.append(nn.ReLU(inplace=self.inplace))

        self.network = nn.Sequential(*network)

        self.mu = nn.Linear(self.hidden_dims[-1], self.nmbr_actions)
        self.sigma = nn.Linear(self.hidden_dims[-1], self.nmbr_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, **kwargs)
        self.to(self.device)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prob = self.network(observation)
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # clamping is computationally cheaper than using an activation like sigmoid
        # adding noise to avoid log(0)
        sigma = torch.clamp(sigma, min=self.noise, max=1)
        return mu, sigma

    def predict(self, observation, reparam: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, sigma = self.forward(observation)
        probs = Normal(mu, sigma)

        if reparam:
            actions = probs.rsample()
        else:
            actions = probs.sample()

        # activate the chosen action and scale it to resemble the environment's scale
        action = torch.tanh(actions) * torch.tensor(self.scaler).to(self.device)
        log_probs = probs.log_prob(actions)  # log, because it is more stable to optimize

        # Appendix C, https://arxiv.org/pdf/1812.05905v2.pdf
        log_probs -= torch.log(1 - action.pow(2) + self.noise)  # to avoid log(0)
        log_probs = log_probs.sum(1, keepdim=True)  # need a scalar quantity for the loss
        return action, log_probs

    def update(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
