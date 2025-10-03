"""
Training Utilities for World Models

This module provides training utilities and trainers for world models,
including VAE, dynamics, reward models, and RSSM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm


class WorldModelTrainer:
    """Trainer for world models"""

    def __init__(
        self,
        world_model: nn.Module,
        learning_rate: float = 1e-3,
        device: torch.device = None,
        beta_schedule: str = 'constant',
        beta_value: float = 1.0
    ):
        self.world_model = world_model
        self.device = device or torch.device('cpu')
        self.world_model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.world_model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        # Training history
        self.loss_history = {
            'total_loss': [],
            'vae_loss': [],
            'dynamics_loss': [],
            'reward_loss': []
        }
        
        # Beta scheduling
        self.beta_schedule = beta_schedule
        self.beta_value = beta_value
        self.step_count = 0

    def get_beta(self) -> float:
        """Get current beta value for VAE loss"""
        if self.beta_schedule == 'constant':
            return self.beta_value
        elif self.beta_schedule == 'linear':
            # Linear increase from 0 to beta_value over 1000 steps
            return min(self.beta_value, self.step_count / 1000.0 * self.beta_value)
        elif self.beta_schedule == 'cyclical':
            # Cyclical beta schedule
            cycle_length = 1000
            cycle_position = self.step_count % cycle_length
            return self.beta_value * (cycle_position / cycle_length)
        else:
            return self.beta_value

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.world_model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        obs = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_obs = batch['next_observations'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        
        # Compute loss
        beta = self.get_beta()
        losses = self.world_model.compute_loss(obs, actions, next_obs, rewards, beta)
        
        # Backward pass
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update step count
        self.step_count += 1
        
        # Store losses
        for key, value in losses.items():
            self.loss_history[key].append(value.item())
        
        return {key: value.item() for key, value in losses.items()}

    def train_epoch(
        self, 
        data_loader: DataLoader, 
        num_batches: int = None
    ) -> Dict[str, float]:
        """Train for one epoch"""
        total_losses = {key: 0.0 for key in self.loss_history.keys()}
        num_batches_processed = 0
        
        for batch in data_loader:
            if num_batches and num_batches_processed >= num_batches:
                break
                
            losses = self.train_step(batch)
            
            for key, value in losses.items():
                total_losses[key] += value
            
            num_batches_processed += 1
        
        # Average losses
        avg_losses = {key: value / num_batches_processed for key, value in total_losses.items()}
        return avg_losses

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate world model"""
        self.world_model.eval()
        total_losses = {key: 0.0 for key in self.loss_history.keys()}
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                obs = batch['observations'].to(self.device)
                actions = batch['actions'].to(self.device)
                next_obs = batch['next_observations'].to(self.device)
                rewards = batch['rewards'].to(self.device)
                
                # Compute loss
                beta = self.get_beta()
                losses = self.world_model.compute_loss(obs, actions, next_obs, rewards, beta)
                
                for key, value in losses.items():
                    total_losses[key] += value.item()
                
                num_batches += 1
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses


class RSSMTrainer:
    """Trainer for Recurrent State Space Models"""

    def __init__(
        self,
        rssm: nn.Module,
        learning_rate: float = 1e-3,
        device: torch.device = None
    ):
        self.rssm = rssm
        self.device = device or torch.device('cpu')
        self.rssm.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.rssm.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)
        
        # Training history
        self.loss_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'reward_loss': [],
            'kl_loss': []
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.rssm.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        obs_seq = batch['observations'].to(self.device)
        action_seq = batch['actions'].to(self.device)
        reward_seq = batch['rewards'].to(self.device)
        
        # Initialize hidden state
        batch_size = obs_seq.shape[0]
        initial_hidden = torch.zeros(batch_size, self.rssm.hidden_dim, device=self.device)
        
        # Compute loss
        losses = self.rssm.compute_loss(obs_seq, action_seq, reward_seq, initial_hidden)
        
        # Backward pass
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Store losses
        for key, value in losses.items():
            self.loss_history[key].append(value.item())
        
        return {key: value.item() for key, value in losses.items()}

    def train_epoch(
        self, 
        data_loader: DataLoader, 
        num_batches: int = None
    ) -> Dict[str, float]:
        """Train for one epoch"""
        total_losses = {key: 0.0 for key in self.loss_history.keys()}
        num_batches_processed = 0
        
        for batch in data_loader:
            if num_batches and num_batches_processed >= num_batches:
                break
                
            losses = self.train_step(batch)
            
            for key, value in losses.items():
                total_losses[key] += value
            
            num_batches_processed += 1
        
        # Average losses
        avg_losses = {key: value / num_batches_processed for key, value in total_losses.items()}
        return avg_losses

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate RSSM"""
        self.rssm.eval()
        total_losses = {key: 0.0 for key in self.loss_history.keys()}
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                obs_seq = batch['observations'].to(self.device)
                action_seq = batch['actions'].to(self.device)
                reward_seq = batch['rewards'].to(self.device)
                
                # Initialize hidden state
                batch_size = obs_seq.shape[0]
                initial_hidden = torch.zeros(batch_size, self.rssm.hidden_dim, device=self.device)
                
                # Compute loss
                losses = self.rssm.compute_loss(obs_seq, action_seq, reward_seq, initial_hidden)
                
                for key, value in losses.items():
                    total_losses[key] += value.item()
                
                num_batches += 1
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses


class VAETrainer:
    """Trainer for Variational Autoencoders"""

    def __init__(
        self,
        vae: nn.Module,
        learning_rate: float = 1e-3,
        device: torch.device = None,
        beta_schedule: str = 'constant',
        beta_value: float = 1.0
    ):
        self.vae = vae
        self.device = device or torch.device('cpu')
        self.vae.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        # Training history
        self.loss_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': []
        }
        
        # Beta scheduling
        self.beta_schedule = beta_schedule
        self.beta_value = beta_value
        self.step_count = 0

    def get_beta(self) -> float:
        """Get current beta value"""
        if self.beta_schedule == 'constant':
            return self.beta_value
        elif self.beta_schedule == 'linear':
            return min(self.beta_value, self.step_count / 1000.0 * self.beta_value)
        else:
            return self.beta_value

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.vae.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        obs = batch.to(self.device)
        
        # Forward pass
        recon_obs, mean, log_var, z = self.vae(obs)
        
        # Compute loss
        beta = self.get_beta()
        total_loss = self.vae.loss_function(recon_obs, obs, mean, log_var, beta)
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update step count
        self.step_count += 1
        
        # Compute individual losses
        recon_loss = torch.nn.functional.mse_loss(recon_obs, obs, reduction='sum')
        kl_loss = self.vae.kl_divergence(mean, log_var)
        
        # Store losses
        self.loss_history['total_loss'].append(total_loss.item())
        self.loss_history['reconstruction_loss'].append(recon_loss.item())
        self.loss_history['kl_loss'].append(kl_loss.item())
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }