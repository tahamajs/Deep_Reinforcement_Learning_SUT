"""
Variational Autoencoder for World Models

This module implements a Variational Autoencoder (VAE) for learning compressed
latent representations of observations in world models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAEEncoder(nn.Module):
    """Variational encoder for world models"""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        # Build encoder network
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output layers for mean and log variance
        self.mean_layer = nn.Linear(prev_dim, latent_dim)
        self.log_var_layer = nn.Linear(prev_dim, latent_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observations to latent space"""
        encoded = self.encoder(obs)
        mean = self.mean_layer(encoded)
        log_var = self.log_var_layer(encoded)
        return mean, log_var

    def sample(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std


class VAEDecoder(nn.Module):
    """Variational decoder for world models"""

    def __init__(self, latent_dim: int, obs_dim: int, hidden_dims: list = [128, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        # Build decoder network
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, obs_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation space"""
        return self.decoder(latent)


class VariationalAutoencoder(nn.Module):
    """Complete VAE for world models"""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.encoder = VAEEncoder(obs_dim, latent_dim, hidden_dims)
        self.decoder = VAEDecoder(latent_dim, obs_dim, hidden_dims[::-1])

    def encode(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observations to latent space"""
        mean, log_var = self.encoder(obs)
        z = self.encoder.sample(mean, log_var)
        return mean, log_var, z

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation"""
        return self.decoder(latent)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE"""
        mean, log_var, z = self.encode(obs)
        recon_obs = self.decode(z)
        return recon_obs, mean, log_var, z

    def loss_function(
        self,
        reconstruction: torch.Tensor,
        obs: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Compute VAE loss (reconstruction + KL divergence)"""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, obs, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return recon_loss + beta * kl_loss

    def kl_divergence(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence"""
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
