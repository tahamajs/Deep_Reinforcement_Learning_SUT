"""
Latent Critic for Planning in Latent Space
"""

import torch
import torch.nn as nn


class LatentCritic(nn.Module):
    """Critic network for latent space value estimation"""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state).squeeze(-1)
