"""
Recurrent State Space Model (RSSM) Implementation

This module implements a Recurrent State Space Model for temporal world modeling
with recurrent neural networks to capture long-term dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RSSM(nn.Module):
    """Recurrent State Space Model for world models"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        stochastic_size: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.stochastic_size = stochastic_size
        self.deterministic_size = latent_dim - stochastic_size

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stochastic_size),  # Posterior parameters
        )

        # Recurrent model (GRU)
        self.rnn = nn.GRUCell(stochastic_size + action_dim, self.deterministic_size)

        # Prior model
        self.prior_net = nn.Sequential(
            nn.Linear(self.deterministic_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stochastic_size),  # Prior parameters
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(stochastic_size + self.deterministic_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
            nn.Sigmoid(),
        )

        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(stochastic_size + self.deterministic_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Continue predictor (for episode termination)
        self.continue_predictor = nn.Sequential(
            nn.Linear(stochastic_size + self.deterministic_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def imagine_step(
        self,
        prev_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of imagination in latent space"""
        # Update deterministic state
        rnn_input = torch.cat([prev_latent, prev_action], dim=-1)
        det_state = self.rnn(rnn_input, prev_state)

        # Sample from prior
        prior_params = self.prior_net(det_state)
        prior_mean, prior_log_var = prior_params.chunk(2, dim=-1)
        prior_latent = self._sample_latent(prior_mean, prior_log_var)

        # Predict reward and continue
        full_state = torch.cat([prior_latent, det_state], dim=-1)
        reward = self.reward_predictor(full_state)
        continue_prob = self.continue_predictor(full_state)

        return det_state, prior_latent, reward, continue_prob

    def observe_step(
        self, obs: torch.Tensor, prev_state: torch.Tensor, prev_action: torch.Tensor, prev_latent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of observation processing"""
        # Update deterministic state
        # If prev_latent is not provided, use zeros (for the first step in a sequence)
        if prev_latent is None:
            rnn_input = torch.cat([torch.zeros_like(prev_state[:, : self.stochastic_size]), prev_action], dim=-1)
        else:
            rnn_input = torch.cat([prev_latent, prev_action], dim=-1)

        det_state = self.rnn(rnn_input, prev_state)

        # Posterior from observation
        posterior_params = self.obs_encoder(obs)
        post_mean, post_log_var = posterior_params.chunk(2, dim=-1)
        post_latent = self._sample_latent(post_mean, post_log_var)

        # Prior for comparison
        prior_params = self.prior_net(det_state)
        prior_mean, prior_log_var = prior_params.chunk(2, dim=-1)

        # Decode observation
        full_state = torch.cat([post_latent, det_state], dim=-1)
        obs_reconstruction = self.decoder(full_state)
        reward_pred = self.reward_predictor(full_state)
        continue_pred = self.continue_predictor(full_state)

        return (
            det_state,
            post_latent,
            obs_reconstruction,
            reward_pred,
            continue_pred,
            prior_mean,
            prior_log_var,
            post_mean,
            post_log_var,
        )

    def _sample_latent(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def imagine_trajectory(
        self,
        initial_state: torch.Tensor,
        initial_latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Imagine trajectory given action sequence"""
        det_states = []
        latents = []
        rewards = []

        state = initial_state
        latent = initial_latent

        for action in actions:
            state, latent, reward, _ = self.imagine_step(state, action, latent)
            det_states.append(state)
            latents.append(latent)
            rewards.append(reward)

        return torch.stack(det_states), torch.stack(latents), torch.stack(rewards)
