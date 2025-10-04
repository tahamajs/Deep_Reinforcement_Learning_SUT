"""
Model definitions for CA07 DQN experiments
==========================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class QNetwork(nn.Module):
    """Standard Q-network for DQN"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Dueling Q-network architecture"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingQNetwork, self).__init__()

        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage: Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def get_value_and_advantage(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get separate value and advantage estimates"""
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        return value, advantage


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """Reset noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)


class NoisyQNetwork(nn.Module):
    """Q-network with noisy layers for exploration"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(NoisyQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.noisy1 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy2 = NoisyLinear(hidden_dim, action_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy1(x))
        return self.noisy2(x)

    def reset_noise(self):
        """Reset noise in noisy layers"""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class CategoricalQNetwork(nn.Module):
    """Categorical Q-network for distributional RL"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super(CategoricalQNetwork, self).__init__()
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim * atoms)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape to (batch_size, action_dim, atoms)
        x = x.view(-1, self.action_dim, self.atoms)

        # Apply softmax to get probabilities
        return F.softmax(x, dim=-1)

    def get_atoms(self) -> torch.Tensor:
        """Get atom values"""
        return torch.linspace(self.v_min, self.v_max, self.atoms)


class RainbowQNetwork(nn.Module):
    """Rainbow Q-network combining multiple improvements"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super(RainbowQNetwork, self).__init__()
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max

        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, atoms),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, action_dim * atoms),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Reshape advantage
        advantage = advantage.view(-1, self.action_dim, self.atoms)

        # Combine value and advantage
        q_values = value.unsqueeze(1) + advantage - advantage.mean(dim=1, keepdim=True)

        # Apply softmax to get probabilities
        return F.softmax(q_values, dim=-1)

    def reset_noise(self):
        """Reset noise in noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def get_atoms(self) -> torch.Tensor:
        """Get atom values"""
        return torch.linspace(self.v_min, self.v_max, self.atoms)


def create_model(
    model_type: str, state_dim: int, action_dim: int, hidden_dim: int = 128, **kwargs
) -> nn.Module:
    """Create a model based on type"""
    if model_type == "q_network":
        return QNetwork(state_dim, action_dim, hidden_dim)
    elif model_type == "dueling":
        return DuelingQNetwork(state_dim, action_dim, hidden_dim)
    elif model_type == "noisy":
        return NoisyQNetwork(state_dim, action_dim, hidden_dim)
    elif model_type == "categorical":
        return CategoricalQNetwork(state_dim, action_dim, hidden_dim, **kwargs)
    elif model_type == "rainbow":
        return RainbowQNetwork(state_dim, action_dim, hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> dict:
    """Get information about a model"""
    return {
        "total_parameters": count_parameters(model),
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters())
        / (1024 * 1024),
        "architecture": str(model),
    }
