"""
Dynamics Models for World Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDynamicsModel(nn.Module):
    """Dynamics model in latent space"""

    def __init__(self, latent_dim, action_dim, hidden_dim=256, stochastic=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.stochastic = stochastic

        # Dynamics network
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if stochastic:
            # Stochastic dynamics: predict mean and log variance
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        else:
            # Deterministic dynamics
            self.fc_next_state = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, a):
        """Predict next latent state given current state and action"""
        # Concatenate state and action
        za = torch.cat([z, a], dim=-1)
        h = self.dynamics(za)

        if self.stochastic:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)

            if self.training:
                # Sample during training
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z_next = mu + eps * std
            else:
                # Use mean during evaluation
                z_next = mu

            return z_next, mu, logvar
        else:
            z_next = self.fc_next_state(h)
            return z_next

    def loss_function(self, z_pred, z_target, mu=None, logvar=None):
        """Dynamics model loss"""
        if self.stochastic and mu is not None and logvar is not None:
            # Stochastic dynamics loss with KL regularization
            pred_loss = F.mse_loss(z_pred, z_target)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return pred_loss + 0.001 * kl_loss  # Small KL weight
        else:
            # Deterministic dynamics loss
            return F.mse_loss(z_pred, z_target)
