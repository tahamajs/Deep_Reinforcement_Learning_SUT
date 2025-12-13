"""
Variational Autoencoder for World Models

This module implements a Variational Autoencoder (VAE) for learning compressed
latent representations of observations in world models.
"""

from .config import VAE_CONFIG
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAEEncoder(nn.Module):
    """Variational encoder for world models"""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),  # Mean and log variance
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent distribution parameters"""
        params = self.encoder(obs)
        mean, log_var = params.chunk(2, dim=-1)
        return mean, log_var

    def sample(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std


class VAEDecoder(nn.Module):
    """Variational decoder for world models"""

    def __init__(self, latent_dim: int, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
            nn.Sigmoid(),  # Assuming normalized observations
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation"""
        return self.decoder(latent)


class VariationalAutoencoder(nn.Module):
    """Complete VAE for world models"""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.encoder = VAEEncoder(obs_dim, latent_dim, hidden_dim)
        self.decoder = VAEDecoder(latent_dim, obs_dim, hidden_dim)

    def encode(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observation and sample latent"""
        mean, log_var = self.encoder(obs)
        latent = self.encoder.sample(mean, log_var)
        return latent, mean, log_var

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation"""
        return self.decoder(latent)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass"""
        latent, mean, log_var = self.encode(obs)
        reconstruction = self.decode(latent)
        return reconstruction, latent, mean, log_var

    def loss_function(
        self,
        reconstruction: torch.Tensor,
        obs: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """VAE loss: reconstruction + KL divergence"""
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, obs, reduction="sum")

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return recon_loss + kl_loss
