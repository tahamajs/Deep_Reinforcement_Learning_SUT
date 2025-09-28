"""
Training utilities for World Models
"""

import torch
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
from .world_model import WorldModel
from .rssm import RecurrentStateSpaceModel


class WorldModelTrainer:
    """Trainer for world model components"""

    def __init__(self, world_model, device, lr=1e-3):
        self.world_model = world_model.to(device)
        self.device = device

        # Optimizers for different components
        self.vae_optimizer = optim.Adam(world_model.vae.parameters(), lr=lr)
        self.dynamics_optimizer = optim.Adam(world_model.dynamics.parameters(), lr=lr)
        self.reward_optimizer = optim.Adam(world_model.reward_model.parameters(), lr=lr)

        # Training statistics
        self.losses = {
            'vae_total': [],
            'vae_recon': [],
            'vae_kl': [],
            'dynamics': [],
            'reward': []
        }

    def train_step(self, batch):
        """Single training step on a batch of data"""
        obs, actions, rewards, next_obs = batch

        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)

        # Train VAE on observations
        self.vae_optimizer.zero_grad()
        recon_obs, mu_obs, logvar_obs, z_obs = self.world_model.vae(obs)
        recon_next_obs, mu_next_obs, logvar_next_obs, z_next_obs = self.world_model.vae(next_obs)

        vae_loss_obs, recon_loss_obs, kl_loss_obs = self.world_model.vae.loss_function(
            obs, recon_obs, mu_obs, logvar_obs)
        vae_loss_next_obs, recon_loss_next_obs, kl_loss_next_obs = self.world_model.vae.loss_function(
            next_obs, recon_next_obs, mu_next_obs, logvar_next_obs)

        vae_total_loss = vae_loss_obs + vae_loss_next_obs
        vae_total_loss.backward()
        self.vae_optimizer.step()

        # Train dynamics model
        self.dynamics_optimizer.zero_grad()
        z_obs_detached = z_obs.detach()
        z_next_obs_detached = z_next_obs.detach()

        if self.world_model.dynamics.stochastic:
            z_pred, mu_pred, logvar_pred = self.world_model.dynamics(z_obs_detached, actions)
            dynamics_loss = self.world_model.dynamics.loss_function(
                z_pred, z_next_obs_detached, mu_pred, logvar_pred)
        else:
            z_pred = self.world_model.dynamics(z_obs_detached, actions)
            dynamics_loss = self.world_model.dynamics.loss_function(z_pred, z_next_obs_detached)

        dynamics_loss.backward()
        self.dynamics_optimizer.step()

        # Train reward model
        self.reward_optimizer.zero_grad()
        pred_rewards = self.world_model.reward_model(z_obs_detached, actions)
        reward_loss = self.world_model.reward_model.loss_function(pred_rewards, rewards)
        reward_loss.backward()
        self.reward_optimizer.step()

        # Record losses
        self.losses['vae_total'].append(vae_total_loss.item())
        self.losses['vae_recon'].append((recon_loss_obs + recon_loss_next_obs).item())
        self.losses['vae_kl'].append((kl_loss_obs + kl_loss_next_obs).item())
        self.losses['dynamics'].append(dynamics_loss.item())
        self.losses['reward'].append(reward_loss.item())

        return {
            'vae_loss': vae_total_loss.item(),
            'dynamics_loss': dynamics_loss.item(),
            'reward_loss': reward_loss.item()
        }


class RSSMTrainer:
    """Trainer for RSSM model"""

    def __init__(self, rssm_model, device, lr=1e-4, kl_weight=1.0, free_nats=3.0):
        self.rssm_model = rssm_model.to(device)
        self.device = device
        self.kl_weight = kl_weight
        self.free_nats = free_nats  # Free nats for KL regularization

        self.optimizer = optim.Adam(rssm_model.parameters(), lr=lr, eps=1e-4)

        # Training statistics
        self.losses = {
            'total': [],
            'reconstruction': [],
            'kl_divergence': [],
            'reward': []
        }

    def kl_divergence(self, post_mean, post_std, prior_mean, prior_std):
        """Compute KL divergence between posterior and prior"""
        post_dist = Normal(post_mean, post_std)
        prior_dist = Normal(prior_mean, prior_std)
        kl = kl_divergence(post_dist, prior_dist)

        # Apply free nats
        kl = torch.maximum(kl, torch.tensor(self.free_nats, device=self.device))

        return kl.sum(-1)  # Sum over stochastic dimensions

    def train_step(self, batch):
        """Single training step"""
        observations, actions, rewards = batch
        batch_size, seq_len = observations.shape[:2]

        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        # Initialize states
        h, z = self.rssm_model.initial_state(batch_size)

        # Storage for losses
        reconstruction_losses = []
        kl_losses = []
        reward_losses = []

        # Forward pass through sequence
        for t in range(seq_len):
            # Observe step
            h, z, (prior_mean, prior_std), (post_mean, post_std) = self.rssm_model.observe(
                observations[:, t], h, z, actions[:, t])

            # Reconstruction loss
            pred_obs = self.rssm_model.decode_obs(h, z)
            recon_loss = torch.mean((pred_obs - observations[:, t]) ** 2)
            reconstruction_losses.append(recon_loss)

            # KL loss
            kl_loss = self.kl_divergence(post_mean, post_std, prior_mean, prior_std)
            kl_losses.append(kl_loss)

            # Reward loss
            pred_reward = self.rssm_model.predict_reward(h, z)
            reward_loss = torch.mean((pred_reward - rewards[:, t]) ** 2)
            reward_losses.append(reward_loss)

        # Aggregate losses
        reconstruction_loss = torch.stack(reconstruction_losses).mean()
        kl_loss = torch.stack(kl_losses).mean()
        reward_loss = torch.stack(reward_losses).mean()

        # Total loss
        total_loss = reconstruction_loss + self.kl_weight * kl_loss + reward_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rssm_model.parameters(), 100.0)
        self.optimizer.step()

        # Record losses
        self.losses['total'].append(total_loss.item())
        self.losses['reconstruction'].append(reconstruction_loss.item())
        self.losses['kl_divergence'].append(kl_loss.item())
        self.losses['reward'].append(reward_loss.item())

        return {
            'total_loss': total_loss.item(),
            'recon_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item(),
            'reward_loss': reward_loss.item()
        }