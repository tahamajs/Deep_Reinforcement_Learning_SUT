import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VariationalWorldModel(nn.Module):
    """Variational Autoencoder-based World Model for Model-Based RL"""

    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=128):
        super(VariationalWorldModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder: obs -> latent mean and log variance
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: latent -> observation reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

        # Dynamics model: (latent, action) -> next latent
        self.dynamics_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and logvar
        )

        # Reward model: (latent, action) -> reward
        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Done model: (latent, action) -> done probability
        self.done_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def encode(self, obs):
        """Encode observation to latent space"""
        h = self.encoder(obs)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation to observation"""
        return self.decoder(z)
    
    def forward(self, obs):
        """Forward pass: encode -> decode"""
        mu, logvar = self.encode(obs)
        z = self.reparameterize(mu, logvar)
        recon_obs = self.decode(z)
        return recon_obs, mu, logvar
    
    def dynamics_forward(self, z, action):
        """Predict next latent state given current latent and action"""
        # Convert action to one-hot if needed
        if action.dim() == 1:
            action_onehot = F.one_hot(action, self.action_dim).float()
        else:
            action_onehot = action.float()
        
        # Concatenate latent and action
        input_vec = torch.cat([z, action_onehot], dim=-1)
        
        # Predict next latent (mean and log variance)
        output = self.dynamics_model(input_vec)
        next_mu = output[:, :self.latent_dim]
        next_logvar = output[:, self.latent_dim:]
        
        return next_mu, next_logvar
    
    def predict_next_latent(self, z, action):
        """Predict next latent state (returns sampled latent)"""
        next_mu, next_logvar = self.dynamics_forward(z, action)
        return self.reparameterize(next_mu, next_logvar)
    
    def predict_reward(self, z, action):
        """Predict reward given latent state and action"""
        # Convert action to one-hot if needed
        if action.dim() == 1:
            action_onehot = F.one_hot(action, self.action_dim).float()
        else:
            action_onehot = action.float()
        
        input_vec = torch.cat([z, action_onehot], dim=-1)
        return self.reward_model(input_vec)
    
    def predict_done(self, z, action):
        """Predict done probability given latent state and action"""
        # Convert action to one-hot if needed
        if action.dim() == 1:
            action_onehot = F.one_hot(action, self.action_dim).float()
        else:
            action_onehot = action.float()
        
        input_vec = torch.cat([z, action_onehot], dim=-1)
        return self.done_model(input_vec)
    
    def compute_loss(self, obs, actions, rewards, next_obs, dones):
        """Compute total world model loss"""
        batch_size = obs.size(0)
        
        # VAE loss (reconstruction + KL divergence)
        recon_obs, mu, logvar = self.forward(obs)
        recon_loss = F.mse_loss(recon_obs, obs, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss + kl_loss
        
        # Encode observations to latent
        z = self.reparameterize(mu, logvar)
        next_mu, next_logvar = self.encode(next_obs)
        next_z = self.reparameterize(next_mu, next_logvar)
        
        # Dynamics loss
        pred_next_mu, pred_next_logvar = self.dynamics_forward(z, actions)
        dynamics_loss = F.mse_loss(pred_next_mu, next_z, reduction='sum')
        
        # Reward loss
        pred_rewards = self.predict_reward(z, actions)
        reward_loss = F.mse_loss(pred_rewards.squeeze(), rewards, reduction='sum')
        
        # Done loss
        pred_dones = self.predict_done(z, actions)
        done_loss = F.binary_cross_entropy(pred_dones.squeeze(), dones.float(), reduction='sum')
        
        # Total loss
        total_loss = vae_loss + dynamics_loss + reward_loss + done_loss
        
        return {
            'total': total_loss,
            'vae': vae_loss,
            'dynamics': dynamics_loss,
            'reward': reward_loss,
            'done': done_loss
        }
    
    def imagine_trajectory(self, z_start, actions, horizon=10):
        """Generate imagined trajectory in latent space"""
        z_trajectory = [z_start]
        rewards = []
        dones = []
        
        z_current = z_start
        for t in range(horizon):
            if t < len(actions):
            action = actions[t]
            else:
                # Random action if not provided
                action = torch.randint(0, self.action_dim, (z_current.size(0),))
            
            # Predict next state
            z_next = self.predict_next_latent(z_current, action)
            reward = self.predict_reward(z_current, action)
            done = self.predict_done(z_current, action)
            
            z_trajectory.append(z_next)
            rewards.append(reward)
            dones.append(done)
            
            z_current = z_next
            
            # Stop if done
            if done.mean() > 0.5:  # If more than half of batch is done
                break
        
        return z_trajectory, rewards, dones
    
    def decode_trajectory(self, z_trajectory):
        """Decode latent trajectory to observations"""
        obs_trajectory = []
        for z in z_trajectory:
            obs = self.decode(z)
            obs_trajectory.append(obs)
        return obs_trajectory


class RecurrentWorldModel(nn.Module):
    """Recurrent version of world model for temporal dependencies"""
    
    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=128, rnn_hidden=64):
        super(RecurrentWorldModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.rnn_hidden = rnn_hidden
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # RNN for temporal dynamics
        self.rnn = nn.GRU(latent_dim + action_dim, rnn_hidden, batch_first=True)
        
        # Output heads
        self.latent_head = nn.Linear(rnn_hidden, latent_dim * 2)  # mean and logvar
        self.reward_head = nn.Linear(rnn_hidden, 1)
        self.done_head = nn.Linear(rnn_hidden, 1)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
    def forward(self, obs_sequence, action_sequence, hidden=None):
        """Process sequence of observations and actions"""
        batch_size, seq_len, _ = obs_sequence.shape
        
        # Encode observations
        obs_flat = obs_sequence.view(-1, self.obs_dim)
        latent_flat = self.encoder(obs_flat)
        latent = latent_flat.view(batch_size, seq_len, self.latent_dim)
        
        # Prepare RNN input
        action_onehot = F.one_hot(action_sequence, self.action_dim).float()
        rnn_input = torch.cat([latent, action_onehot], dim=-1)
        
        # RNN forward
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        
        # Predictions
        latent_output = self.latent_head(rnn_output)
        next_mu = latent_output[:, :, :self.latent_dim]
        next_logvar = latent_output[:, :, self.latent_dim:]
        
        rewards = self.reward_head(rnn_output)
        dones = torch.sigmoid(self.done_head(rnn_output))
        
        return {
            'next_mu': next_mu,
            'next_logvar': next_logvar,
            'rewards': rewards,
            'dones': dones,
            'hidden': hidden
        }