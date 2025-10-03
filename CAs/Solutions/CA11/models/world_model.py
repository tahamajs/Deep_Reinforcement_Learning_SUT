"""
World Model Implementation

This module implements a complete world model that combines VAE, dynamics,
and reward models for model-based reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from .vae import VariationalAutoencoder
from .dynamics import LatentDynamicsModel
from .reward_model import RewardModel


class WorldModel(nn.Module):
    """Complete world model combining VAE, dynamics, and reward models"""

    def __init__(
        self,
        vae: VariationalAutoencoder,
        dynamics: LatentDynamicsModel,
        reward_model: RewardModel,
    ):
        super().__init__()
        self.vae = vae
        self.dynamics = dynamics
        self.reward_model = reward_model

        # Store dimensions
        self.obs_dim = vae.obs_dim
        self.latent_dim = vae.latent_dim
        self.action_dim = dynamics.action_dim

    def encode_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent space"""
        _, _, z = self.vae.encode(obs)
        return z

    def decode_observations(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent states to observations"""
        return self.vae.decode(latent)

    def predict_next_state(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next latent state"""
        next_latent, _, _ = self.dynamics(latent, action)
        return next_latent

    def predict_reward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict reward"""
        return self.reward_model(latent, action)

    def predict_next_state_and_reward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next observation and reward from current observation"""
        # Encode current observation
        latent = self.encode_observations(obs)

        # Predict next latent state
        next_latent = self.predict_next_state(latent, action)

        # Decode next observation
        next_obs = self.decode_observations(next_latent)

        # Predict reward
        reward = self.predict_reward(latent, action)

        return next_obs, reward

    def imagine_trajectory(
        self, initial_obs: torch.Tensor, actions: torch.Tensor, horizon: int = 10
    ) -> Dict[str, torch.Tensor]:
        """Imagine a trajectory using the world model"""
        batch_size = initial_obs.shape[0]
        device = initial_obs.device

        # Initialize trajectory
        observations = [initial_obs]
        latents = [self.encode_observations(initial_obs)]
        rewards = []

        current_latent = latents[0]

        for t in range(horizon):
            if t < actions.shape[1]:
                action = actions[:, t]
            else:
                # Use zero action if not enough actions provided
                action = torch.zeros(batch_size, self.action_dim, device=device)

            # Predict next state and reward
            next_latent = self.predict_next_state(current_latent, action)
            reward = self.predict_reward(current_latent, action)
            next_obs = self.decode_observations(next_latent)

            # Store results
            observations.append(next_obs)
            latents.append(next_latent)
            rewards.append(reward)

            # Update current state
            current_latent = next_latent

        return {
            "observations": torch.stack(
                observations, dim=1
            ),  # [batch, horizon+1, obs_dim]
            "latents": torch.stack(latents, dim=1),  # [batch, horizon+1, latent_dim]
            "rewards": torch.stack(rewards, dim=1),  # [batch, horizon]
        }

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute world model loss"""
        # VAE loss
        recon_obs, mean, log_var, z = self.vae(obs)
        vae_loss = self.vae.loss_function(recon_obs, obs, mean, log_var, beta)

        # Dynamics loss
        next_latent_pred, mean_dyn, log_var_dyn = self.dynamics(z, actions)
        _, _, z_next_true = self.vae.encode(next_obs)

        if self.dynamics.stochastic:
            # KL divergence for stochastic dynamics
            kl_dyn = -0.5 * torch.sum(
                1 + log_var_dyn - mean_dyn.pow(2) - log_var_dyn.exp()
            )
            dynamics_loss = F.mse_loss(next_latent_pred, z_next_true) + 0.1 * kl_dyn
        else:
            dynamics_loss = F.mse_loss(next_latent_pred, z_next_true)

        # Reward loss
        reward_pred = self.reward_model(z, actions)
        reward_loss = F.mse_loss(reward_pred, rewards)

        # Total loss
        total_loss = vae_loss + dynamics_loss + reward_loss

        return {
            "total_loss": total_loss,
            "vae_loss": vae_loss,
            "dynamics_loss": dynamics_loss,
            "reward_loss": reward_loss,
        }

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through world model"""
        return self.imagine_trajectory(obs, actions.unsqueeze(1), horizon=1)
