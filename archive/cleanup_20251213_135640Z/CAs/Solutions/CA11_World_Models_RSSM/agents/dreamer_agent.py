"""
Dreamer Agent for Planning in Latent Space

This module implements the Dreamer agent, a model-based reinforcement learning
algorithm that learns world models and plans in latent space for sample-efficient
reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .latent_actor import LatentActor
from .latent_critic import LatentCritic
from ..experiments.config import AGENT_CONFIG, DREAMER_CONFIG, GLOBAL_CONFIG
from ..models.rssm import RSSM
from collections import deque
import random


class LatentActorCritic(nn.Module):
    """Complete latent actor-critic"""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor = LatentActor(latent_dim, action_dim, hidden_dim)
        self.critic = LatentCritic(latent_dim, hidden_dim)

    def act(self, latent: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select action in latent space"""
        if deterministic:
            mean, _ = self.actor(latent)
            return torch.tanh(mean)
        else:
            action, _ = self.actor.sample(latent)
            return action

    def evaluate(
        self, latents: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions in latent space"""
        mean, log_std = self.actor(latents)
        std = torch.exp(log_std)

        normal = Normal(mean, std)
        log_prob = normal.log_prob(actions).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        values = self.critic(latents)

        return log_prob, values


class DreamerAgent:
    """Complete Dreamer agent implementation"""

    def __init__(
        self, obs_dim: int, action_dim: int, global_config: Any
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = global_config.dreamer_config.agent_config.latent_dim
        self.imagination_horizon = global_config.dreamer_config.agent_config.imagination_horizon
        self.device = global_config.device

        # World model components
        self.rssm = RSSM(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            latent_dim=global_config.dreamer_config.rssm_config.latent_dim,
            hidden_dim=global_config.dreamer_config.rssm_config.hidden_dim,
            stochastic_size=global_config.dreamer_config.rssm_config.stochastic_size,
        ).to(self.device)

        # Actor-critic in latent space
        self.actor_critic = LatentActorCritic(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=global_config.dreamer_config.agent_config.hidden_dim,
        ).to(self.device)

        # Experience buffer
        self.buffer = deque(maxlen=100000)

        # Optimizers
        self.world_optimizer = optim.Adam(self.rssm.parameters(), lr=global_config.dreamer_config.rssm_config.learning_rate)
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=global_config.dreamer_config.agent_config.actor_lr)
        self.critic_optimizer = optim.Adam(
            self.actor_critic.critic.parameters(), lr=global_config.dreamer_config.agent_config.critic_lr
        )

    def select_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Select action using current world model"""
        # Encode observation
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            post_params = self.rssm.obs_encoder(obs_tensor)
            latent = self.rssm._sample_latent(*post_params.chunk(2, dim=-1))

            # Get action from actor
            action = self.actor_critic.act(latent, deterministic)

        return action.squeeze(0).cpu().numpy()

    def store_transition(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
    ):
        """Store transition in buffer"""
        self.buffer.append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
            }
        )

    def update_world_model(self, batch_size: int) -> Dict[str, float]:
        """Update world model using data from buffer"""
        if len(self.buffer) < batch_size:
            return {}

        # Sample batch
        batch = random.sample(list(self.buffer), batch_size)
        obs_batch = torch.tensor(np.array([t["obs"] for t in batch]), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(np.array([t["action"] for t in batch]), dtype=torch.float32).to(self.device)
        reward_batch = torch.tensor([t["reward"] for t in batch], dtype=torch.float32).to(self.device)
        # next_obs_batch = torch.tensor(np.array([t["next_obs"] for t in batch]), dtype=torch.float32).to(self.device)

        self.world_optimizer.zero_grad()

        # Process sequence (simplified single-step for now)
        det_state = torch.zeros(batch_size, self.rssm.deterministic_size, device=self.device)
        prev_latent = torch.zeros(batch_size, self.rssm.stochastic_size, device=self.device)

        # Observation step
        (
            det_state,
            post_latent,
            obs_recon,
            reward_pred,
            continue_pred,
            prior_mean,
            prior_log_var,
            post_mean,
            post_log_var,
        ) = self.rssm.observe_step(obs_batch, det_state, action_batch, prev_latent)

        # Losses
        obs_loss = F.mse_loss(obs_recon, obs_batch)
        reward_loss = F.mse_loss(reward_pred.squeeze(), reward_batch)
        continue_loss = F.binary_cross_entropy(
            continue_pred.squeeze(),
            torch.tensor([not t["done"] for t in batch], dtype=torch.float32, device=self.device),
        )

        # KL divergence between posterior and prior
        kl_loss = self._kl_divergence(
            post_mean, post_log_var, prior_mean, prior_log_var
        )

        total_loss = obs_loss + reward_loss + continue_loss + kl_loss

        total_loss.backward()
        self.world_optimizer.step()

        return {
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_world_loss": total_loss.item(),
        }

    def _kl_divergence(
        self,
        mean1: torch.Tensor,
        log_var1: torch.Tensor,
        mean2: torch.Tensor,
        log_var2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between two Gaussians"""
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)

        kl = 0.5 * (log_var2 - log_var1 + (var1 + (mean1 - mean2).pow(2)) / var2 - 1)
        return kl.sum()

    def update_actor_critic(self, batch_size: int) -> Dict[str, float]:
        """Update actor-critic using imagination"""
        if len(self.buffer) < batch_size:
            return {}

        # Sample initial states from buffer
        batch = random.sample(list(self.buffer), batch_size)
        obs_batch = torch.tensor(np.array([t["obs"] for t in batch]), dtype=torch.float32).to(self.device)

        # Encode initial observations
        with torch.no_grad():
            post_params = self.rssm.obs_encoder(obs_batch)
            init_latent = self.rssm._sample_latent(*post_params.chunk(2, dim=-1))
            init_det_state = torch.zeros(batch_size, self.rssm.deterministic_size, device=self.device)

        # Imagine trajectories
        imagined_latents = []
        imagined_rewards = []
        imagined_actions = []
        imagined_log_probs = []

        latent = init_latent
        det_state = init_det_state

        for _ in range(self.imagination_horizon):
            # Sample action
            action, log_prob = self.actor_critic.actor.sample(latent)

            # Imagine next state
            det_state, latent, reward, _ = self.rssm.imagine_step(
                det_state, action, latent
            )

            imagined_latents.append(latent)
            imagined_rewards.append(reward)
            imagined_actions.append(action)
            imagined_log_probs.append(log_prob)

        # Stack imagined trajectory
        imagined_latents = torch.stack(imagined_latents)  # [horizon, batch, latent_dim]
        imagined_rewards = torch.stack(imagined_rewards)  # [horizon, batch, 1]
        imagined_actions = torch.stack(imagined_actions)  # [horizon, batch, action_dim]
        imagined_log_probs = torch.stack(imagined_log_probs)  # [horizon, batch, 1]

        # Compute returns
        returns = self._compute_returns(imagined_rewards, gamma=AGENT_CONFIG.gamma)

        # Update critic
        self.critic_optimizer.zero_grad()
        values = self.actor_critic.critic(imagined_latents.view(-1, self.latent_dim))
        critic_loss = F.mse_loss(values, returns.view(-1, 1))
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        advantages = returns - values.detach()
        actor_loss = -(imagined_log_probs.view(-1) * advantages.view(-1)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()
        }

    def _compute_returns(
        self, rewards: torch.Tensor, gamma: float = 0.99
    ) -> torch.Tensor:
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.stack(returns)
