"""
Latent Critic for Planning in Latent Space
"""

from ..experiments.config import AGENT_CONFIG
import torch
import torch.nn as nn


class LatentCritic(nn.Module):
    """Critic for latent space value estimation"""

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate value of latent state"""
        return self.critic(latent)
