from typing import Optional
import torch
import torch.nn as nn


class MLPBase(nn.Module):
    """Simple MLP backbone used by policy and value networks."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, input_dim)
        Returns:
            features: (batch, hidden_dim)
        """
        return self.net(x)


class MLPPolicy(nn.Module):
    """
    A small stochastic policy producing action logits and sampling methods.
    Suitable for discrete actions.
    """

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.backbone = MLPBase(input_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns action logits (unnormalized).
        Shape: (batch, action_dim)
        """
        features = self.backbone(x)
        return self.logits(features)

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample actions given observations.
        Returns:
            actions: (batch,)
        """
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        return dist.sample()

    def log_prob(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return log probabilities of selected actions (batch,)."""
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions)


class MLPValue(nn.Module):
    """A value-function approximator returning scalar values per observation."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = MLPBase(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns value estimates with shape (batch,).
        """
        v = self.backbone(x)
        return self.head(v).squeeze(-1)











