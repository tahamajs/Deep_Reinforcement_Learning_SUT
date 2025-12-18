from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """Simple MLP policy suitable for discrete action spaces.

    The module is import-safe (no heavy work at import time) and typed.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.logits = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits (unnormalized)."""
        return self.logits(self.net(x))

    def act(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample an action or return argmax if deterministic.

        Args:
            x: observation tensor, shape (batch, obs_dim) or (obs_dim,)
            deterministic: when True, use argmax
        Returns:
            action tensor (int64)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            return torch.argmax(probs, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

    def get_action_dist(self, x: torch.Tensor) -> torch.distributions.Categorical:
        """Return a Categorical distribution for the given observations."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs)








