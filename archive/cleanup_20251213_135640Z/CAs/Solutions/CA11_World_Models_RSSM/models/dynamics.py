"""
Dynamics Model for World Models

This module implements latent dynamics models for predicting next states
in the latent space of world models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LatentDynamicsModel(nn.Module):
    """Dynamics model for latent space transitions"""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 128],
        stochastic: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.stochastic = stochastic

        # Build dynamics network
        layers = []
        prev_dim = latent_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        self.dynamics_net = nn.Sequential(*layers)

        if stochastic:
            # Output layers for mean and log variance
            self.mean_layer = nn.Linear(prev_dim, latent_dim)
            self.log_var_layer = nn.Linear(prev_dim, latent_dim)
        else:
            # Deterministic output
            self.output_layer = nn.Linear(prev_dim, latent_dim)

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through dynamics model

        Args:
            latent: Current latent state [batch, latent_dim]
            action: Action taken [batch, action_dim]

        Returns:
            next_latent: Predicted next latent state
            mean: Mean of predicted distribution (if stochastic)
            log_var: Log variance of predicted distribution (if stochastic)
        """
        # Combine latent and action
        combined = torch.cat([latent, action], dim=-1)

        # Forward through dynamics network
        dynamics_out = self.dynamics_net(combined)

        if self.stochastic:
            # Sample from predicted distribution
            mean = self.mean_layer(dynamics_out)
            log_var = self.log_var_layer(dynamics_out)
            log_var = torch.clamp(log_var, -20, 2)  # Clamp for numerical stability

            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            next_latent = mean + eps * std

            return next_latent, mean, log_var
        else:
            # Deterministic prediction
            next_latent = self.output_layer(dynamics_out)
            return next_latent, None, None

    def predict_deterministic(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next latent state deterministically"""
        if not self.stochastic:
            return self.forward(latent, action)[0]

        # For stochastic model, use mean
        combined = torch.cat([latent, action], dim=-1)
        dynamics_out = self.dynamics_net(combined)
        mean = self.mean_layer(dynamics_out)
        return mean

    def sample_next_state(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Sample next latent state from predicted distribution"""
        if self.stochastic:
            return self.forward(latent, action)[0]
        else:
            return self.predict_deterministic(latent, action)

    def compute_kl_divergence(
        self, mean: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence with standard normal"""
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
