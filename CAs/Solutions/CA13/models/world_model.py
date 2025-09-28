"""World Model implementations for model-based reinforcement learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class VariationalWorldModel(nn.Module):
    """Variational Autoencoder-based world model for learning environment dynamics."""

    def __init__(self, obs_dim, action_dim, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder (observation -> latent distribution)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        # Dynamics model (latent state + action -> next latent distribution)
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Dynamics outputs (mean and log variance)
        self.dynamics_mu = nn.Linear(hidden_dim, latent_dim)
        self.dynamics_logvar = nn.Linear(hidden_dim, latent_dim)

        # Reward model
        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Decoder (latent state -> observation)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

    def encode(self, obs):
        """Encode observation to latent distribution parameters."""
        h = self.encoder(obs)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def dynamics_forward(self, latent_state, action):
        """Predict next latent state given current state and action."""
        # One-hot encode action if discrete
        if len(action.shape) == 1:
            action_one_hot = F.one_hot(action.long(), self.action_dim).float()
        else:
            action_one_hot = action

        dynamics_input = torch.cat([latent_state, action_one_hot], dim=-1)
        h = self.dynamics(dynamics_input)

        mu = self.dynamics_mu(h)
        logvar = self.dynamics_logvar(h)

        return mu, logvar

    def predict_reward(self, latent_state, action):
        """Predict reward given latent state and action."""
        if len(action.shape) == 1:
            action_one_hot = F.one_hot(action.long(), self.action_dim).float()
        else:
            action_one_hot = action

        reward_input = torch.cat([latent_state, action_one_hot], dim=-1)
        return self.reward_model(reward_input)

    def decode(self, latent_state):
        """Decode latent state to observation."""
        return self.decoder(latent_state)

    def forward(self, obs, action=None):
        """Full forward pass through world model."""
        # Encode observation
        mu_enc, logvar_enc = self.encode(obs)
        latent_state = self.reparameterize(mu_enc, logvar_enc)

        # Decode observation (reconstruction)
        recon_obs = self.decode(latent_state)

        results = {
            'latent_state': latent_state,
            'mu_enc': mu_enc,
            'logvar_enc': logvar_enc,
            'recon_obs': recon_obs
        }

        # If action provided, predict dynamics and reward
        if action is not None:
            mu_dyn, logvar_dyn = self.dynamics_forward(latent_state, action)
            next_latent = self.reparameterize(mu_dyn, logvar_dyn)
            pred_reward = self.predict_reward(latent_state, action)

            results.update({
                'mu_dyn': mu_dyn,
                'logvar_dyn': logvar_dyn,
                'next_latent': next_latent,
                'pred_reward': pred_reward
            })

        return results

    def imagine_trajectory(self, initial_obs, actions):
        """Imagine trajectory given initial observation and action sequence."""
        batch_size = initial_obs.shape[0]
        sequence_length = len(actions)

        # Encode initial observation
        mu_enc, logvar_enc = self.encode(initial_obs)
        current_latent = self.reparameterize(mu_enc, logvar_enc)

        # Store trajectory
        trajectory = {
            'latent_states': [current_latent],
            'observations': [self.decode(current_latent)],
            'rewards': [],
            'actions': []
        }

        # Roll out trajectory
        for t in range(sequence_length):
            action = actions[t]
            trajectory['actions'].append(action)

            # Predict reward
            pred_reward = self.predict_reward(current_latent, action)
            trajectory['rewards'].append(pred_reward)

            # Predict next latent state
            mu_dyn, logvar_dyn = self.dynamics_forward(current_latent, action)
            next_latent = self.reparameterize(mu_dyn, logvar_dyn)

            # Update current state
            current_latent = next_latent
            trajectory['latent_states'].append(current_latent)
            trajectory['observations'].append(self.decode(current_latent))

        return trajectory


class WorldModelLoss:
    """Loss functions for training world models."""

    def __init__(self, recon_weight=1.0, kl_weight=1.0, reward_weight=1.0, dynamics_weight=1.0):
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.reward_weight = reward_weight
        self.dynamics_weight = dynamics_weight

    def reconstruction_loss(self, recon_obs, target_obs):
        """Reconstruction loss between predicted and actual observations."""
        return F.mse_loss(recon_obs, target_obs)

    def kl_divergence_loss(self, mu, logvar):
        """KL divergence loss for VAE regularization."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

    def reward_loss(self, pred_reward, target_reward):
        """Reward prediction loss."""
        return F.mse_loss(pred_reward.squeeze(), target_reward)

    def dynamics_loss(self, pred_next_latent, target_next_latent):
        """Dynamics prediction loss in latent space."""
        return F.mse_loss(pred_next_latent, target_next_latent)

    def compute_total_loss(self, model_output, target_obs, target_reward=None, target_next_obs=None):
        """Compute total loss for world model training."""
        losses = {}

        # Reconstruction loss
        recon_loss = self.reconstruction_loss(model_output['recon_obs'], target_obs)
        losses['reconstruction'] = recon_loss

        # KL divergence loss
        kl_loss = self.kl_divergence_loss(model_output['mu_enc'], model_output['logvar_enc'])
        losses['kl_divergence'] = kl_loss

        # Total loss starts with reconstruction and KL
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss

        # Reward loss (if target reward provided)
        if target_reward is not None and 'pred_reward' in model_output:
            reward_loss = self.reward_loss(model_output['pred_reward'], target_reward)
            losses['reward'] = reward_loss
            total_loss += self.reward_weight * reward_loss

        # Dynamics loss (if target next observation provided)
        if target_next_obs is not None and 'mu_dyn' in model_output:
            # Encode target next observation to latent space
            with torch.no_grad():
                target_mu, _ = model_output['mu_enc'], model_output['logvar_enc']  # Placeholder - need next obs encoding

            dynamics_loss = F.mse_loss(model_output['mu_dyn'], model_output['next_latent'])
            losses['dynamics'] = dynamics_loss
            total_loss += self.dynamics_weight * dynamics_loss

        losses['total'] = total_loss
        return losses