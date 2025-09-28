"""
Recurrent State Space Models (RSSM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class RecurrentStateSpaceModel(nn.Module):
    """Recurrent State Space Model (RSSM) for temporal world modeling"""

    def __init__(
        self, obs_dim, action_dim, stoch_dim=30, deter_dim=200, hidden_dim=400
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim  # Stochastic state dimension
        self.deter_dim = deter_dim  # Deterministic state dimension
        self.hidden_dim = hidden_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),  # Mean and std
        )

        self.rnn = nn.GRUCell(stoch_dim + action_dim, deter_dim)

        self.transition_model = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),  # Mean and std
        )

        self.representation_model = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),  # Mean and std
        )

        self.obs_decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        self.reward_model = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.value_model = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def initial_state(self, batch_size):
        """Initialize hidden state"""
        h = torch.zeros(
            batch_size, self.deter_dim, device=next(self.parameters()).device
        )
        z = torch.zeros(
            batch_size, self.stoch_dim, device=next(self.parameters()).device
        )
        return h, z

    def encode_obs(self, obs):
        """Encode observation to stochastic state distribution"""
        encoded = self.obs_encoder(obs)
        mean, std = torch.chunk(encoded, 2, dim=-1)
        std = F.softplus(std) + 1e-4
        return mean, std

    def transition_prior(self, h):
        """Prior transition model p(z_t | h_t)"""
        encoded = self.transition_model(h)
        mean, std = torch.chunk(encoded, 2, dim=-1)
        std = F.softplus(std) + 1e-4
        return mean, std

    def representation_posterior(self, h, obs_encoded):
        """Posterior representation model q(z_t | h_t, o_t)"""
        encoded = self.representation_model(torch.cat([h, obs_encoded], dim=-1))
        mean, std = torch.chunk(encoded, 2, dim=-1)
        std = F.softplus(std) + 1e-4
        return mean, std

    def reparameterize(self, mean, std):
        """Reparameterization trick"""
        if self.training:
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def recurrent_step(self, prev_h, prev_z, action):
        """Single recurrent step"""
        rnn_input = torch.cat([prev_z, action], dim=-1)
        h = self.rnn(rnn_input, prev_h)
        return h

    def observe(self, obs, prev_h, prev_z, action):
        """Observation step: encode observation and update state"""
        h = self.recurrent_step(prev_h, prev_z, action)

        obs_encoded = self.obs_encoder(obs)

        prior_mean, prior_std = self.transition_prior(h)
        post_mean, post_std = self.representation_posterior(h, obs_encoded)

        z = self.reparameterize(post_mean, post_std)

        return h, z, (prior_mean, prior_std), (post_mean, post_std)

    def imagine(self, prev_h, prev_z, action):
        """Imagination step: predict next state without observation"""
        h = self.recurrent_step(prev_h, prev_z, action)

        prior_mean, prior_std = self.transition_prior(h)

        z = self.reparameterize(prior_mean, prior_std)

        return h, z, (prior_mean, prior_std)

    def decode_obs(self, h, z):
        """Decode observation from state"""
        state = torch.cat([h, z], dim=-1)
        return self.obs_decoder(state)

    def predict_reward(self, h, z):
        """Predict reward from state"""
        state = torch.cat([h, z], dim=-1)
        return self.reward_model(state).squeeze(-1)

    def predict_value(self, h, z):
        """Predict value from state"""
        state = torch.cat([h, z], dim=-1)
        return self.value_model(state).squeeze(-1)
