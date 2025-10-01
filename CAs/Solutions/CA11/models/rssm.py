"""
Recurrent State Space Model (RSSM) Implementation

This module implements a Recurrent State Space Model for temporal world modeling
with recurrent neural networks to capture long-term dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class RSSM(nn.Module):
    """Recurrent State Space Model for world models"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        stochastic_size: int = 32,
        rnn_type: str = 'gru'
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.stochastic_size = stochastic_size
        self.deter_dim = hidden_dim
        
        # Encoder for observations
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder for observations
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim)
        )
        
        # Recurrent network
        if rnn_type == 'gru':
            self.rnn = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Stochastic state components
        self.stoch_mean = nn.Linear(hidden_dim, stochastic_size)
        self.stoch_std = nn.Linear(hidden_dim, stochastic_size)
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Done predictor (optional)
        self.done_predictor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _sample_stochastic(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Sample from stochastic state distribution"""
        eps = torch.randn_like(mean)
        return mean + std * eps

    def _encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent representation"""
        return self.encoder(obs)

    def _decode_observation(self, latent: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Decode latent and hidden state to observation"""
        combined = torch.cat([latent, hidden], dim=-1)
        return self.decoder(combined)

    def imagine_step(
        self,
        prev_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single imagination step"""
        # Combine previous latent and action
        rnn_input = torch.cat([prev_latent, prev_action], dim=-1).unsqueeze(1)
        
        # Update hidden state
        _, new_hidden = self.rnn(rnn_input, prev_state.unsqueeze(0))
        new_hidden = new_hidden.squeeze(0)
        
        # Sample new stochastic state
        stoch_mean = self.stoch_mean(new_hidden)
        stoch_std = F.softplus(self.stoch_std(new_hidden)) + 0.1
        new_latent = self._sample_stochastic(stoch_mean, stoch_std)
        
        # Predict next observation
        next_obs = self._decode_observation(new_latent, new_hidden)
        
        # Predict reward
        combined = torch.cat([new_latent, new_hidden], dim=-1)
        reward = self.reward_predictor(combined).squeeze(-1)
        
        return next_obs, reward, new_hidden

    def observe_step(
        self, obs: torch.Tensor, prev_state: torch.Tensor, prev_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single observation step"""
        # Encode observation
        encoded_obs = self._encode_observation(obs)
        
        # Combine with previous action
        rnn_input = torch.cat([encoded_obs, prev_action], dim=-1).unsqueeze(1)
        
        # Update hidden state
        _, new_hidden = self.rnn(rnn_input, prev_state.unsqueeze(0))
        new_hidden = new_hidden.squeeze(0)
        
        # Sample stochastic state
        stoch_mean = self.stoch_mean(new_hidden)
        stoch_std = F.softplus(self.stoch_std(new_hidden)) + 0.1
        new_latent = self._sample_stochastic(stoch_mean, stoch_std)
        
        # Reconstruct observation
        recon_obs = self._decode_observation(new_latent, new_hidden)
        
        # Predict reward
        combined = torch.cat([new_latent, new_hidden], dim=-1)
        reward = self.reward_predictor(combined).squeeze(-1)
        
        return new_latent, new_hidden, recon_obs, reward, stoch_mean

    def imagine_trajectory(
        self,
        initial_state: torch.Tensor,
        initial_latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Imagine a complete trajectory"""
        batch_size, horizon = actions.shape[:2]
        device = actions.device
        
        # Initialize trajectory
        observations = []
        rewards = []
        hidden_states = [initial_state]
        latents = [initial_latent]
        
        current_state = initial_state
        current_latent = initial_latent
        
        for t in range(horizon):
            action = actions[:, t]
            
            # Imagination step
            next_obs, reward, next_state = self.imagine_step(
                current_state, action, current_latent
            )
            
            # Store results
            observations.append(next_obs)
            rewards.append(reward)
            hidden_states.append(next_state)
            
            # Update for next step
            current_state = next_state
            current_latent = self._sample_stochastic(
                self.stoch_mean(next_state),
                F.softplus(self.stoch_std(next_state)) + 0.1
            )
            latents.append(current_latent)
        
        return (
            torch.stack(observations, dim=1),  # [batch, horizon, obs_dim]
            torch.stack(rewards, dim=1),       # [batch, horizon]
            torch.stack(hidden_states[1:], dim=1)  # [batch, horizon, hidden_dim]
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for single step"""
        return self.imagine_step(obs, actions, hidden)

    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        initial_hidden: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute RSSM loss"""
        batch_size, seq_len = obs_seq.shape[:2]
        device = obs_seq.device
        
        # Initialize
        current_hidden = initial_hidden
        total_recon_loss = 0
        total_reward_loss = 0
        total_kl_loss = 0
        
        for t in range(seq_len):
            obs_t = obs_seq[:, t]
            action_t = action_seq[:, t]
            reward_t = reward_seq[:, t]
            
            # Observation step
            latent_t, hidden_t, recon_obs_t, pred_reward_t, stoch_mean = self.observe_step(
                obs_t, current_hidden, action_t
            )
            
            # Compute losses
            recon_loss = F.mse_loss(recon_obs_t, obs_t)
            reward_loss = F.mse_loss(pred_reward_t, reward_t)
            
            # KL divergence loss (if using stochastic states)
            kl_loss = -0.5 * torch.sum(1 + torch.log(stoch_mean.pow(2) + 1e-8) - stoch_mean.pow(2))
            
            total_recon_loss += recon_loss
            total_reward_loss += reward_loss
            total_kl_loss += kl_loss
            
            current_hidden = hidden_t
        
        # Average losses
        avg_recon_loss = total_recon_loss / seq_len
        avg_reward_loss = total_reward_loss / seq_len
        avg_kl_loss = total_kl_loss / seq_len
        
        total_loss = avg_recon_loss + avg_reward_loss + 0.1 * avg_kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': avg_recon_loss,
            'reward_loss': avg_reward_loss,
            'kl_loss': avg_kl_loss
        }