"""
Training Utilities for World Models

This module provides training utilities and trainers for world models,
RSSM, and related components.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from tqdm import tqdm


class WorldModelTrainer:
    """Trainer for world models"""

    def __init__(
        self,
        world_model: nn.Module,
        learning_rate: float = 1e-3,
        device: torch.device = None,
    ):
        self.world_model = world_model
        self.device = device or torch.device("cpu")
        self.optimizer = optim.Adam(world_model.parameters(), lr=learning_rate)

        self.losses = {
            "total": [],
            "vae": [],
            "dynamics": [],
            "reward": [],
        }

    def train_epoch(
        self, dataloader, beta: float = 1.0, clip_grad_norm: float = 1.0
    ) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_losses = {"total": 0, "vae": 0, "dynamics": 0, "reward": 0}

        self.world_model.train()

        for batch in dataloader:
            # Move batch to device
            obs = batch["observations"].to(self.device)
            actions = batch["actions"].to(self.device)
            next_obs = batch["next_observations"].to(self.device)
            rewards = batch["rewards"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            losses = self.world_model.compute_loss(
                obs, actions, next_obs, rewards, beta
            )

            # Backward pass
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.world_model.parameters(), clip_grad_norm
            )
            self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()

        # Average losses
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            self.losses[key].append(epoch_losses[key])

        return epoch_losses

    def train(
        self, dataloader, epochs: int = 100, beta: float = 1.0, print_interval: int = 10
    ) -> Dict[str, List[float]]:
        """Train the world model"""
        print(f"Training world model for {epochs} epochs...")

        for epoch in tqdm(range(epochs)):
            epoch_losses = self.train_epoch(dataloader, beta)

            if (epoch + 1) % print_interval == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Total Loss = {epoch_losses['total']:.4f}, "
                    f"VAE Loss = {epoch_losses['vae']:.4f}, "
                    f"Dynamics Loss = {epoch_losses['dynamics']:.4f}, "
                    f"Reward Loss = {epoch_losses['reward']:.4f}"
                )

        return self.losses


class RSSMTrainer:
    """Trainer for RSSM models"""

    def __init__(
        self,
        rssm: nn.Module,
        learning_rate: float = 1e-3,
        device: torch.device = None,
    ):
        self.rssm = rssm
        self.device = device or torch.device("cpu")
        self.optimizer = optim.Adam(rssm.parameters(), lr=learning_rate)

        self.losses = {
            "total": [],
            "reconstruction": [],
            "reward": [],
            "kl": [],
        }

    def train_epoch(
        self, dataloader, kl_weight: float = 0.1, clip_grad_norm: float = 1.0
    ) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_losses = {"total": 0, "reconstruction": 0, "reward": 0, "kl": 0}

        self.rssm.train()

        for batch in dataloader:
            # Move batch to device
            obs_seq = batch["observations"].to(self.device)
            action_seq = batch["actions"].to(self.device)
            reward_seq = batch["rewards"].to(self.device)

            batch_size = obs_seq.shape[0]
            initial_hidden = torch.zeros(batch_size, self.rssm.deter_dim).to(
                self.device
            )

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            losses = self.rssm.compute_loss(
                obs_seq, action_seq, reward_seq, initial_hidden
            )

            # Adjust KL weight
            losses["kl_loss"] = kl_weight * losses["kl_loss"]
            losses["total_loss"] = (
                losses["reconstruction_loss"]
                + losses["reward_loss"]
                + losses["kl_loss"]
            )

            # Backward pass
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), clip_grad_norm)
            self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()

        # Average losses
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            self.losses[key].append(epoch_losses[key])

        return epoch_losses

    def train(
        self,
        dataloader,
        epochs: int = 100,
        kl_weight: float = 0.1,
        print_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """Train the RSSM"""
        print(f"Training RSSM for {epochs} epochs...")

        for epoch in tqdm(range(epochs)):
            epoch_losses = self.train_epoch(dataloader, kl_weight)

            if (epoch + 1) % print_interval == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Total Loss = {epoch_losses['total']:.4f}, "
                    f"Reconstruction Loss = {epoch_losses['reconstruction']:.4f}, "
                    f"Reward Loss = {epoch_losses['reward']:.4f}, "
                    f"KL Loss = {epoch_losses['kl']:.4f}"
                )

        return self.losses


class VAETrainer:
    """Trainer for VAE models"""

    def __init__(
        self,
        vae: nn.Module,
        learning_rate: float = 1e-3,
        device: torch.device = None,
    ):
        self.vae = vae
        self.device = device or torch.device("cpu")
        self.optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

        self.losses = {
            "total": [],
            "reconstruction": [],
            "kl": [],
        }

    def train_epoch(
        self, dataloader, beta: float = 1.0, clip_grad_norm: float = 1.0
    ) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_losses = {"total": 0, "reconstruction": 0, "kl": 0}

        self.vae.train()

        for batch in dataloader:
            # Move batch to device
            obs = batch.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            recon_obs, mean, log_var, z = self.vae(obs)

            # Compute loss
            recon_loss = F.mse_loss(recon_obs, obs, reduction="sum")
            kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            total_loss = recon_loss + beta * kl_loss

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), clip_grad_norm)
            self.optimizer.step()

            # Accumulate losses
            epoch_losses["total"] += total_loss.item()
            epoch_losses["reconstruction"] += recon_loss.item()
            epoch_losses["kl"] += kl_loss.item()

        # Average losses
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            self.losses[key].append(epoch_losses[key])

        return epoch_losses

    def train(
        self, dataloader, epochs: int = 100, beta: float = 1.0, print_interval: int = 10
    ) -> Dict[str, List[float]]:
        """Train the VAE"""
        print(f"Training VAE for {epochs} epochs...")

        for epoch in tqdm(range(epochs)):
            epoch_losses = self.train_epoch(dataloader, beta)

            if (epoch + 1) % print_interval == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Total Loss = {epoch_losses['total']:.4f}, "
                    f"Reconstruction Loss = {epoch_losses['reconstruction']:.4f}, "
                    f"KL Loss = {epoch_losses['kl']:.4f}"
                )

        return self.losses
