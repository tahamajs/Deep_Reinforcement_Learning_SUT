"""Neural network models for policy and value functions.

Clean, well-typed PyTorch modules with light input validation and helpful
utilities for sampling and evaluating policies.
"""
from __future__ import annotations

from typing import Sequence, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes: Sequence[int], activation: Callable = nn.ReLU) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """Categorical policy network for discrete action spaces.

    The forward pass returns logits; use ``get_action`` to sample or obtain
    deterministic actions along with their log-probabilities.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Callable = nn.ReLU,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive integers")
        sizes = [obs_dim, *hidden_sizes, action_dim]
        self.net = _mlp(sizes, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape (..., action_dim).

        Accepts both single observations (..., obs_dim) and batches (batch, obs_dim).
        """
        if x.ndim < 1:
            raise ValueError("Input tensor must have at least 1 dimension")
        logits = self.net(x)
        return logits

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action, log_prob) for input observations.

        If ``x`` is a single observation (obs_dim,), it is treated as a batch of
        size 1 and returned values are squeezed appropriately (action scalar,
        log_prob scalar). For a batch input, batched tensors are returned.
        """
        squeeze_single = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_single = True

        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = probs.argmax(dim=-1)
            logp = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            logp = dist.log_prob(action)

        if squeeze_single:
            return action.squeeze(0), logp.squeeze(0)
        return action, logp


class ValueNetwork(nn.Module):
    """Value function approximator returning a scalar per state."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Callable = nn.ReLU,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be a positive integer")
        sizes = [obs_dim, *hidden_sizes, 1]
        self.net = _mlp(sizes, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return value estimates with shape (...,).

        Input can be a single observation (obs_dim,) or a batch (batch, obs_dim).
        Output is squeezed to remove the final singleton dimension.
        """
        v = self.net(x)
        return v.squeeze(-1)
