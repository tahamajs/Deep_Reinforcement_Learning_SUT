"""
Reward Model for World Models

This module implements reward prediction models for world models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RewardModel(nn.Module):
    """Reward prediction model for world models"""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list = [128, 64],
        output_dim: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.output_dim = output_dim

        # Build reward prediction network
        layers = []
        prev_dim = latent_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.reward_net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through reward model

        Args:
            latent: Current latent state [batch, latent_dim]
            action: Action taken [batch, action_dim]

        Returns:
            reward: Predicted reward [batch, output_dim]
        """
        # Combine latent and action
        combined = torch.cat([latent, action], dim=-1)

        # Predict reward
        reward = self.reward_net(combined)

        return reward

    def predict_reward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict reward for given state and action"""
        return self.forward(latent, action)

    def compute_loss(
        self, latent: torch.Tensor, action: torch.Tensor, true_reward: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss for reward prediction"""
        pred_reward = self.forward(latent, action)
        return F.mse_loss(pred_reward, true_reward)
