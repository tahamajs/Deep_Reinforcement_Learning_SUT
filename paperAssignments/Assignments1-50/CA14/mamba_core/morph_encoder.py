"""Morphology encoder module.

Provides a GRU-based encoder that consumes sequences of (obs, act, rew, done)
and returns a reparameterized latent z_morph with (mu, logvar).
"""

from typing import Tuple

import torch
import torch.nn as nn


class MorphEncoder(nn.Module):
    """GRU-based morphology encoder.

    Args:
        obs_dim: dimensionality of observations per timestep
        act_dim: dimensionality of actions per timestep
        latent_dim: output latent dimensionality for morphology
        hidden: hidden size of GRU
    """

    def __init__(
        self, obs_dim: int, act_dim: int, latent_dim: int, hidden: int = 256
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(obs_dim + act_dim + 2, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a batch of sequences into morphology latent.

        Shapes:
            obs: (B, T, obs_dim)
            act: (B, T, act_dim)
            rew: (B, T)
            done: (B, T)

        Returns:
            z: (B, latent_dim)
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        x = torch.cat([obs, act, rew.unsqueeze(-1), done.unsqueeze(-1)], dim=-1)
        _, h = self.rnn(x)
        h = h[-1]
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


