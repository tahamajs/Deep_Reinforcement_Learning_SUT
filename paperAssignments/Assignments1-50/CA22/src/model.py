from __future__ import annotations
from typing import Tuple
import torch
from torch import nn


class PolicyNet(nn.Module):
    """Simple MLP policy producing action logits for discrete actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return unnormalized logits of shape (batch, action_dim)."""
        return self.net(obs)


class ValueNet(nn.Module):
    """Value function approximator returning scalar values."""

    def __init__(self, obs_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return value estimates of shape (batch,)."""
        v = self.net(obs)
        return v.view(-1)


def sample_action(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample discrete action from logits, return (action, log_prob).

    logits: (batch, action_dim)
    action: (batch,)
    log_prob: (batch,)
    """
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob








