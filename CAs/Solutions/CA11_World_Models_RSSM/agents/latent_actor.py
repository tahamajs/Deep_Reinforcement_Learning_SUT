"""
Latent Actor for Planning in Latent Space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from ..experiments.config import AGENT_CONFIG
from typing import Tuple


class LatentActor(nn.Module):
    """Actor for latent space policy"""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim, 2 * action_dim
            ),  # Mean and log std for continuous actions
        )

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action distribution parameters"""
        params = self.actor(latent)
        mean, log_std = params.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)  # Clamp for numerical stability
        return mean, log_std

    def sample(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std = self.forward(latent)
        std = torch.exp(log_std)

        normal = Normal(mean, std)
        action = normal.rsample()

        # Compute log probability
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)

        # Squash action to [-1, 1]
        action = torch.tanh(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob
