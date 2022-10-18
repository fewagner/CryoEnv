import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from ._utils import init_weights


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

        self.apply(init_weights)
        self.to(self.device)

    def forward(self, observations):
        x = self.network(observations)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

    def sample(self, observations):
        mu, log_std = self.forward(observations)
        sigma = log_std.exp()

        probs = Normal(mu, sigma)
        sample = probs.rsample()  # apply the reparam trick
        actions = torch.tanh(sample)

        log_probs = probs.log_prob(sample)
        log_probs -= torch.log(1 - actions.pow(2) + self.noise)
        log_probs = log_probs.sum(dim=1, keepdim=True)  # scalar value needed for loss

        return actions, log_probs
