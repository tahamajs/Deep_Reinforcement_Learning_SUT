"""Actor network with FiLM conditioning from morphology latent.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FilmLayer(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.scale = nn.Linear(cond_dim, in_dim)
        self.shift = nn.Linear(cond_dim, in_dim)

    def forward(self, x: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        return self.scale(zc) * x + self.shift(zc)


class Actor(nn.Module):
    """Gaussian actor conditioned on morphology via FiLM.

    Usage:
        mu, std = actor(z, z_m)
        action = mu + std * eps
    """

    def __init__(self, latent_dim: int, morph_dim: int, hidden: int = 256, act_dim: int = 6) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + morph_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.film = FilmLayer(hidden, morph_dim)
        self.mu = nn.Linear(hidden, act_dim)
        self.logstd = nn.Linear(hidden, act_dim)

    def forward(self, z: torch.Tensor, z_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, z_m], -1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.film(x, z_m)
        mu = self.mu(x)
        logstd = torch.clamp(self.logstd(x), -5.0, 2.0)
        std = torch.exp(logstd)
        return mu, std

    def sample(self, z: torch.Tensor, z_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, std = self.forward(z, z_m)
        eps = torch.randn_like(std)
        return mu + eps * std, mu

    def act(self, z: torch.Tensor, z_m: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mu, std = self.forward(z, z_m)
        if deterministic:
            return mu
        eps = torch.randn_like(std)
        return mu + eps * std
