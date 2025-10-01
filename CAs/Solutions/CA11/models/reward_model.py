"""
Reward Models for World Models

This module implements reward prediction models that predict rewards
in latent space given current latent states and actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RewardModel(nn.Module):
    """Reward prediction model for world models"""

    def __init__(
        self, 
        latent_dim: int, 
        action_dim: int, 
        hidden_dims: list = [128, 64],
        stochastic: bool = False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.stochastic = stochastic
        
        # Build reward network
        layers = []
        prev_dim = latent_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        if stochastic:
            # Output mean and log variance for stochastic rewards
            self.mean_layer = nn.Linear(prev_dim, 1)
            self.log_var_layer = nn.Linear(prev_dim, 1)
        else:
            # Output deterministic reward
            self.output_layer = nn.Linear(prev_dim, 1)
        
        self.network = nn.Sequential(*layers)

    def forward(
        self, 
        latent: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict reward"""
        x = torch.cat([latent, action], dim=-1)
        features = self.network(x)
        
        if self.stochastic:
            mean = self.mean_layer(features)
            log_var = self.log_var_layer(features)
            log_var = torch.clamp(log_var, -10, 2)
            
            # Sample reward
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            reward = mean + eps * std
            
            return reward.squeeze(-1)
        else:
            reward = self.output_layer(features)
            return reward.squeeze(-1)

    def predict_deterministic(
        self, 
        latent: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict deterministic reward (use mean for stochastic models)"""
        if self.stochastic:
            x = torch.cat([latent, action], dim=-1)
            features = self.network(x)
            mean = self.mean_layer(features)
            return mean.squeeze(-1)
        else:
            return self.forward(latent, action)


class EnsembleRewardModel(nn.Module):
    """Ensemble of reward models for uncertainty estimation"""

    def __init__(
        self, 
        latent_dim: int, 
        action_dim: int, 
        hidden_dims: list = [128, 64],
        num_models: int = 5,
        stochastic: bool = False
    ):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([
            RewardModel(latent_dim, action_dim, hidden_dims, stochastic)
            for _ in range(num_models)
        ])

    def forward(
        self, 
        latent: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict rewards using ensemble"""
        predictions = []
        for model in self.models:
            pred = model.forward(latent, action)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, batch]
        mean_pred = predictions.mean(dim=0)
        
        return mean_pred

    def predict_with_uncertainty(
        self, 
        latent: torch.Tensor, 
        action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict rewards with uncertainty estimates"""
        predictions = []
        for model in self.models:
            pred = model.forward(latent, action)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, batch]
        mean_pred = predictions.mean(dim=0)
        var_pred = predictions.var(dim=0)
        
        return mean_pred, var_pred