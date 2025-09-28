"""
Complete World Model Implementation
"""

import torch
import torch.nn as nn
from .vae import VariationalAutoencoder
from .dynamics import LatentDynamicsModel
from .reward_model import RewardModel


class WorldModel(nn.Module):
    """Complete World Model combining VAE, dynamics, and reward models"""

    def __init__(self, obs_dim, action_dim, latent_dim=64, hidden_dim=256,
                 stochastic_dynamics=True, beta=1.0):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Component models
        self.vae = VariationalAutoencoder(obs_dim, latent_dim, hidden_dim, beta)
        self.dynamics = LatentDynamicsModel(latent_dim, action_dim, hidden_dim, stochastic_dynamics)
        self.reward_model = RewardModel(latent_dim, action_dim, hidden_dim)

        # Training statistics
        self.training_stats = {
            'vae_loss': [],
            'dynamics_loss': [],
            'reward_loss': [],
            'total_loss': []
        }

    def encode_observations(self, obs):
        """Encode observations to latent states"""
        with torch.no_grad():
            mu, logvar = self.vae.encode(obs)
            z = self.vae.reparameterize(mu, logvar)
            return z

    def decode_latent_states(self, z):
        """Decode latent states to observations"""
        with torch.no_grad():
            return self.vae.decode(z)

    def predict_next_state(self, z, a):
        """Predict next latent state"""
        return self.dynamics(z, a)

    def predict_reward(self, z, a):
        """Predict reward"""
        return self.reward_model(z, a)

    def rollout(self, initial_obs, actions, return_observations=False):
        """Perform rollout in world model"""
        batch_size = initial_obs.shape[0]
        horizon = actions.shape[1]

        # Encode initial observation
        z = self.encode_observations(initial_obs)

        states = [z]
        rewards = []
        observations = []

        for t in range(horizon):
            # Predict reward
            r = self.predict_reward(z, actions[:, t])
            rewards.append(r)

            # Predict next state
            if self.dynamics.stochastic:
                z, _, _ = self.predict_next_state(z, actions[:, t])
            else:
                z = self.predict_next_state(z, actions[:, t])

            states.append(z)

            if return_observations:
                obs = self.decode_latent_states(z)
                observations.append(obs)

        results = {
            'states': torch.stack(states, dim=1),
            'rewards': torch.stack(rewards, dim=1)
        }

        if return_observations:
            results['observations'] = torch.stack(observations, dim=1)

        return results