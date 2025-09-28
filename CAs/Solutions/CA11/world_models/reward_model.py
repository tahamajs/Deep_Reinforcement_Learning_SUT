"""
Reward Models for World Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """Reward prediction model in latent space"""

    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.reward_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, a):
        """Predict reward given latent state and action"""
        za = torch.cat([z, a], dim=-1)
        return self.reward_net(za).squeeze(-1)

    def loss_function(self, pred_reward, target_reward):
        """Reward prediction loss"""
        return F.mse_loss(pred_reward, target_reward)
