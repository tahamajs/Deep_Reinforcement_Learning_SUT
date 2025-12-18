from __future__ import annotations
from typing import Sequence, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """Simple MLP policy for discrete actions that returns logits."""

    def __init__(
        self, obs_dim: int, hidden_sizes: Sequence[int], action_dim: int
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden_sizes, action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # returns logits over discrete actions
        return self.net(obs)


class ValueNetwork(nn.Module):
    """Simple MLP critic producing a scalar value per observation."""

    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden_sizes, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (64, 64)
    ) -> None:
        super().__init__()
        self.policy = PolicyNetwork(obs_dim, hidden_sizes, action_dim)
        self.value = ValueNetwork(obs_dim, hidden_sizes)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(obs)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value(obs)
        return logp, entropy, value












