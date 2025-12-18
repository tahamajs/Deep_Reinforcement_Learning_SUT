"""Simplified RSSM-based world model with morphology conditioning.

This module provides a lightweight RSSM (deterministic GRU + gaussian stochastic state)
and small decoder heads for observation and reward prediction. It is intentionally
compact but functional for unit tests and integration with the rest of the assignment.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    """A minimal RSSM with GRU deterministic core and gaussian stochastic latent.

    Args:
        obs_dim: observation dimensionality (used for decoder)
        act_dim: action dimensionality
        deter_dim: deterministic hidden size
        stoch_dim: stochastic latent size
        morph_dim: morphology latent size (conditioning)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        deter_dim: int = 200,
        stoch_dim: int = 32,
        morph_dim: int = 16,
    ) -> None:
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.gru = nn.GRUCell(stoch_dim + act_dim + morph_dim, deter_dim)
        self.prior_net = nn.Linear(deter_dim, stoch_dim * 2)
        self.post_net = nn.Linear(deter_dim + obs_dim, stoch_dim * 2)
        # decoders
        self.obs_decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim + morph_dim, 256),
            nn.ELU(),
            nn.Linear(256, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim + morph_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.discount_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim + morph_dim, 64), nn.ELU(), nn.Linear(64, 1)
        )

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.prior_net(h)
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -10.0, 2.0)
        return mu, logvar

    def posterior(
        self, h: torch.Tensor, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.post_net(torch.cat([h, obs], dim=-1))
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -10.0, 2.0)
        return mu, logvar

    def observe_step(
        self, h: torch.Tensor, a: torch.Tensor, z_m: torch.Tensor, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step observe: update h using previous z,a and compute posterior over z.
        Returns new (h, z, mu_post, logvar_post)
        """
        # GRU input expects (stoch_dim + act_dim + morph_dim).
        # Use previous stochastic z if available; here we don't have z in the signature,
        # so use zeros as a placeholder for previous z to keep shapes consistent.
        prev_z_placeholder = h.new_zeros(h.size(0), self.stoch_dim)
        gru_in = torch.cat([prev_z_placeholder, a, z_m], dim=-1)
        h_new = self.gru(gru_in, h)
        mu_post, logvar_post = self.posterior(h_new, obs)
        std = torch.exp(0.5 * logvar_post)
        eps = torch.randn_like(std)
        z = mu_post + eps * std
        return h_new, z, mu_post, logvar_post

    def imagine_step(
        self, h: torch.Tensor, a: torch.Tensor, z_m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # GRU input expects (stoch_dim + act_dim + morph_dim).
        # For imagination we also lack previous stochastic z, so use zeros as placeholder.
        prev_z_placeholder = h.new_zeros(h.size(0), self.stoch_dim)
        gru_in = torch.cat([prev_z_placeholder, a, z_m], dim=-1)
        h_new = self.gru(gru_in, h)
        mu_prior, logvar_prior = self.prior(h_new)
        std = torch.exp(0.5 * logvar_prior)
        eps = torch.randn_like(std)
        z = mu_prior + eps * std
        return h_new, z, mu_prior, logvar_prior

    def decode_obs(
        self, h: torch.Tensor, z: torch.Tensor, z_m: torch.Tensor
    ) -> torch.Tensor:
        feat = torch.cat([h, z, z_m], dim=-1)
        return self.obs_decoder(feat)

    def predict_reward(
        self, h: torch.Tensor, z: torch.Tensor, z_m: torch.Tensor
    ) -> torch.Tensor:
        feat = torch.cat([h, z, z_m], dim=-1)
        return self.reward_head(feat).squeeze(-1)

    def predict_discount(
        self, h: torch.Tensor, z: torch.Tensor, z_m: torch.Tensor
    ) -> torch.Tensor:
        feat = torch.cat([h, z, z_m], dim=-1)
        return torch.sigmoid(self.discount_head(feat)).squeeze(-1)


class WorldModel(nn.Module):
    """Wrapper exposing simple API for training and imagining."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        deter_dim: int = 200,
        stoch_dim: int = 32,
        morph_dim: int = 16,
    ) -> None:
        super().__init__()
        self.rssm = RSSM(obs_dim, act_dim, deter_dim, stoch_dim, morph_dim)

    def init_state(self, batch_size: int, device: torch.device):
        return self.rssm.init_state(batch_size, device)

    def observe(
        self, h: torch.Tensor, a: torch.Tensor, z_m: torch.Tensor, obs: torch.Tensor
    ):
        return self.rssm.observe_step(h, a, z_m, obs)

    def imagine(self, h: torch.Tensor, a: torch.Tensor, z_m: torch.Tensor):
        return self.rssm.imagine_step(h, a, z_m)

    def decode_obs(
        self, h: torch.Tensor, z: torch.Tensor, z_m: torch.Tensor
    ) -> torch.Tensor:
        return self.rssm.decode_obs(h, z, z_m)

    def predict_reward(
        self, h: torch.Tensor, z: torch.Tensor, z_m: torch.Tensor
    ) -> torch.Tensor:
        return self.rssm.predict_reward(h, z, z_m)

    def predict_discount(
        self, h: torch.Tensor, z: torch.Tensor, z_m: torch.Tensor
    ) -> torch.Tensor:
        return self.rssm.predict_discount(h, z, z_m)








