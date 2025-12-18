from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """Simple Gaussian policy with MLP backbone."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # parametrized log std (state-independent)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: tensor of shape (B, obs_dim)
        Returns:
            mean: (B, action_dim)
            log_std: (action_dim,) broadcastable
        """
        h = self.net(obs)
        mean = self.mean_head(h)
        return mean, self.log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        action = mean + eps * std
        log_prob = -0.5 * (
            ((action - mean) / (std + 1e-8)) ** 2
            + 2 * log_std
            + torch.log(2 * torch.pi)
        )
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)

