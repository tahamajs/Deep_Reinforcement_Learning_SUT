import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    """Standard feed-forward Deep Q-Network."""

    def __init__(
        self, state_size: int, action_size: int, hidden_size: int = 64
    ) -> None:
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQN(nn.Module):
    """Dueling DQN architecture: separate streams for state-value and advantage."""

    def __init__(
        self, state_size: int, action_size: int, hidden_size: int = 64
    ) -> None:
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_value = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_value


class NoisyLinear(nn.Module):
    """A simple Noisy linear layer (independent Gaussian parameter noise)."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))

        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        bound = 1 / np.sqrt(in_features)
        nn.init.uniform_(self.mu_weight, -bound, bound)
        nn.init.uniform_(self.mu_bias, -bound, bound)
        nn.init.constant_(self.sigma_weight, sigma_init)
        nn.init.constant_(self.sigma_bias, sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps_w = torch.randn_like(self.sigma_weight)
            eps_b = torch.randn_like(self.sigma_bias)
            weight = self.mu_weight + self.sigma_weight * eps_w
            bias = self.mu_bias + self.sigma_bias * eps_b
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)


class NoisyDQN(nn.Module):
    """DQN using NoisyLinear layers for exploration (simple two hidden layers)."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(NoisyDQN, self).__init__()
        self.fc1 = NoisyLinear(state_size, hidden_size)
        self.fc2 = NoisyLinear(hidden_size, hidden_size)
        self.fc3 = NoisyLinear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
