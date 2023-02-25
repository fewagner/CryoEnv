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


class LSTMQNetwork(nn.Module):
    # TODO! not tested yet

    def __init__(self, n_observations, n_actions, hidden_size=256, num_layers=2, device="cpu"):
        super(LSTMQNetwork, self).__init__()
        self.device = device
        network_1 = []
        network_2 = []

        input_channels = n_observations + n_actions

        network_1.append(nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True))
        network_2.append(nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True))

        network_1.append(nn.Linear(hidden_size, 1))
        network_2.append(nn.Linear(hidden_size, 1))

        self.network_1 = nn.Sequential(*network_1)
        self.network_2 = nn.Sequential(*network_2)

        self.apply(init_weights)
        self.h0, self.c0 = torch.zeros((num_layers, hidden_size)), torch.zeros((num_layers, hidden_size))
        self.to(self.device)

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], 1)

        x1 = self.network_1(x)
        x2 = self.network_2(x)

        return x1, x2

    def next_step(self, observations, actions):
        x = torch.cat([observations, actions], 1)
        x = x.resize(1, 1, -1)  # batch, sequence, features

        x1, (self.h0, self.c0) = self.network_1(x, (self.h0, self.c0))
        x2, (self.h0, self.c0) = self.network_2(x, (self.h0, self.c0))

        return x1.flatten(), x2.flatten()

    def reset(self):
        self.h0, self.c0 = torch.zeros(self.h0.shape()), torch.zeros(self.c0.shape())