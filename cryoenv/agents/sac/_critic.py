import torch
import torch.nn as nn

from ._utils import init_weights


class QNetwork(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_dims=[256, 256], device="cpu"):
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

        self.apply(init_weights)
        self.to(self.device)

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], 1)

        x1 = self.network_1(x)
        x2 = self.network_2(x)

        return x1, x2
