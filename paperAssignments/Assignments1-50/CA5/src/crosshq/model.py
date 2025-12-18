from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp_block(in_dim: int, out_dim: int, bn_momentum: float):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, momentum=bn_momentum),
        nn.ReLU(),
    )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        depth: int = 3,
        bn_momentum: float = 0.01,
    ):
        super().__init__()
        layers = []
        curr = input_dim
        for _ in range(depth):
            layers.append(_mlp_block(curr, hidden_dim, bn_momentum))
            curr = hidden_dim
        layers.append(nn.Linear(curr, 1))
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossQCritic(nn.Module):
    """Double critic with BatchNorm and CrossQ-friendly forward."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        depth: int = 3,
        bn_momentum: float = 0.01,
    ):
        super().__init__()
        self.input_dim = state_dim + action_dim
        self.q1 = MLP(self.input_dim, hidden_dim, depth, bn_momentum)
        self.q2 = MLP(self.input_dim, hidden_dim, depth, bn_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns concatenated [q1, q2] each (batch, 1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

    def q1_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q1(x)

    def q2_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q2(x)


class GaussianPolicy(nn.Module):
    """Gaussian actor producing rsample and log_prob for continuous actions.

    Note: This is a minimal, production-ready module intended for integration into the CrossHQ
    training skeleton. It uses diagonal Gaussian with squashed tanh actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 512,
        depth: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        layers = []
        curr = obs_dim
        for _ in range(depth):
            layers.append(nn.Linear(curr, hidden))
            layers.append(nn.ReLU())
            curr = hidden
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(curr, action_dim)
        self.logstd_head = nn.Linear(curr, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mu = self.mu_head(h)
        log_std = self.logstd_head(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def dist(self, obs: torch.Tensor):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        return torch.distributions.Normal(mu, std)

    def rsample_and_logprob(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return squashed action and its log-probability (sum over action dim)."""
        dist = self.dist(obs)
        x = dist.rsample()
        logp = dist.log_prob(x).sum(-1, keepdim=True)
        # squash via tanh
        action = torch.tanh(x)
        # log prob correction for tanh (from SAC)
        logp = logp - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, logp

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            mu, _ = self.forward(obs)
            if deterministic:
                return torch.tanh(mu)
            else:
                a, _ = self.rsample_and_logprob(obs)
                return a

