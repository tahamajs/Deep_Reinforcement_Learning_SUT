from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple fully-connected MLP helper.

    The MLP consists of n_layers linear layers with ReLU activations between them
    (no activation on the final layer).
    """

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int = 128, n_layers: int = 2
    ):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU())
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape (B, in_dim)
        Returns:
            output tensor of shape (B, out_dim)
        """
        return self.net(x)


class MLPPolicy(nn.Module):
    """Simple MLP policy producing action logits (for discrete actions)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.logits_net = MLP(obs_dim, action_dim, hidden_dim, n_layers=3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return raw logits for a batch of observations.

        Args:
            obs: (B, obs_dim)
        Returns:
            logits: (B, action_dim)
        """
        return self.logits_net(obs)

    def action_distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(obs)
        return torch.distributions.Categorical(logits=logits)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample or select action and return logprob.

        Returns:
            action: (B,)
            logp: (B,)
        """
        dist = self.action_distribution(obs)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp


class MLPValue(nn.Module):
    """Simple MLP state-value network."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.value_net = MLP(obs_dim, 1, hidden_dim, n_layers=3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # returns shape (B,)
        return self.value_net(obs).squeeze(-1)















