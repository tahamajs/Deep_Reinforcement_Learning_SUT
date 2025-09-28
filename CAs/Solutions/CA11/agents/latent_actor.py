"""
Latent Actor for Planning in Latent Space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class LatentActor(nn.Module):
    """Actor network for latent space planning"""

    def __init__(self, state_dim, action_dim, hidden_dim=256, action_range=1.0):
        super().__init__()
        self.action_range = action_range

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std
        )

        # Initialize last layer with small weights
        self.network[-1].weight.data.uniform_(-1e-3, 1e-3)
        self.network[-1].bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, state):
        output = self.network(state)
        mean, log_std = torch.chunk(output, 2, dim=-1)

        # Constrain log_std
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        return mean, std

    def sample(self, state):
        """Sample action from policy"""
        mean, std = self.forward(state)
        normal = Normal(mean, std)

        # Reparameterization trick
        x = normal.rsample()
        action = torch.tanh(x) * self.action_range

        # Compute log probability
        log_prob = normal.log_prob(x).sum(dim=-1)
        # Correct for tanh transformation
        log_prob -= (2 * (np.log(2) - x - F.softplus(-2 * x))).sum(dim=-1)

        return action, log_prob

    def get_action(self, state, deterministic=False):
        """Get action (used for evaluation)"""
        mean, std = self.forward(state)

        if deterministic:
            action = torch.tanh(mean) * self.action_range
            return action
        else:
            normal = Normal(mean, std)
            x = normal.sample()
            action = torch.tanh(x) * self.action_range
            return action