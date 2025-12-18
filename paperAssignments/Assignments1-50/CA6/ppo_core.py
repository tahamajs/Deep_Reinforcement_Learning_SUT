from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Simple actor-critic supporting continuous (Box) and discrete (Discrete) action spaces.
    The policy head for continuous actions outputs mean and log_std parameter (state-independent std).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        continuous: bool = True,
    ):
        super().__init__()
        self.continuous = continuous
        self.shared = MLP(obs_dim, hidden_sizes, hidden_sizes[-1])
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        if continuous:
            self.mu_head = nn.Linear(hidden_sizes[-1], act_dim)
            # state-independent log std
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        else:
            self.logits = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action_repr, value). For continuous action_repr is mean; for discrete logits."""
        h = self.shared(obs)
        value = self.value_head(h).squeeze(-1)
        if self.continuous:
            mu = self.mu_head(h)
            return mu, value
        else:
            logits = self.logits(h)
            return logits, value

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and return (action, logp, value)
        """
        with torch.no_grad():
            act_repr, value = self.forward(obs)
            if self.continuous:
                std = torch.exp(self.log_std)
                dist = Normal(act_repr, std)
                if deterministic:
                    action = act_repr
                else:
                    action = dist.sample()
                logp = dist.log_prob(action).sum(-1)
                entropy = dist.entropy().sum(-1)
            else:
                dist = Categorical(logits=act_repr)
                if deterministic:
                    action = torch.argmax(act_repr, dim=-1)
                else:
                    action = dist.sample()
                logp = dist.log_prob(action)
                entropy = dist.entropy()
        return action, logp, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities, entropy and value for given (obs, actions) - used in PPO updates.
        Returns (logp, entropy, value)
        """
        act_repr, value = self.forward(obs)
        if self.continuous:
            std = torch.exp(self.log_std)
            dist = Normal(act_repr, std)
            logp = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            dist = Categorical(logits=act_repr)
            logp = dist.log_prob(actions)
            entropy = dist.entropy()
        return logp, entropy, value
