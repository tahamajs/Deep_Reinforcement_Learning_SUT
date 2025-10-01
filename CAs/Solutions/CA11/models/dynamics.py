"""
Latent Dynamics Models for World Models

This module implements dynamics models that predict next latent states
given current latent states and actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LatentDynamicsModel(nn.Module):
    """Dynamics model for predicting next latent states"""

    def __init__(
        self, 
        latent_dim: int, 
        action_dim: int, 
        hidden_dims: list = [256, 128],
        stochastic: bool = True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.stochastic = stochastic
        
        # Build dynamics network
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
            # Output mean and log variance for stochastic dynamics
            self.mean_layer = nn.Linear(prev_dim, latent_dim)
            self.log_var_layer = nn.Linear(prev_dim, latent_dim)
        else:
            # Output deterministic next state
            self.output_layer = nn.Linear(prev_dim, latent_dim)
        
        self.network = nn.Sequential(*layers)

    def forward(
        self, 
        latent: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Predict next latent state"""
        x = torch.cat([latent, action], dim=-1)
        features = self.network(x)
        
        if self.stochastic:
            mean = self.mean_layer(features)
            log_var = self.log_var_layer(features)
            log_var = torch.clamp(log_var, -10, 2)  # Prevent numerical instability
            
            # Sample next state
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            next_latent = mean + eps * std
            
            return next_latent, mean, log_var
        else:
            next_latent = self.output_layer(features)
            return next_latent, None, None

    def predict_deterministic(
        self, 
        latent: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict deterministic next state (use mean for stochastic models)"""
        if self.stochastic:
            x = torch.cat([latent, action], dim=-1)
            features = self.network(x)
            mean = self.mean_layer(features)
            return mean
        else:
            return self.forward(latent, action)[0]


class EnsembleDynamicsModel(nn.Module):
    """Ensemble of dynamics models for uncertainty estimation"""

    def __init__(
        self, 
        latent_dim: int, 
        action_dim: int, 
        hidden_dims: list = [256, 128],
        num_models: int = 5,
        stochastic: bool = True
    ):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([
            LatentDynamicsModel(latent_dim, action_dim, hidden_dims, stochastic)
            for _ in range(num_models)
        ])

    def forward(
        self, 
        latent: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next states using ensemble"""
        predictions = []
        for model in self.models:
            pred = model.forward(latent, action)[0]
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, batch, latent_dim]
        mean_pred = predictions.mean(dim=0)
        var_pred = predictions.var(dim=0)
        
        return mean_pred, var_pred

    def sample_predictions(
        self, 
        latent: torch.Tensor, 
        action: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """Sample predictions from ensemble"""
        all_predictions = []
        
        for _ in range(num_samples):
            model_idx = torch.randint(0, self.num_models, (1,)).item()
            pred = self.models[model_idx].forward(latent, action)[0]
            all_predictions.append(pred)
        
        return torch.stack(all_predictions, dim=0)  # [num_samples, batch, latent_dim]