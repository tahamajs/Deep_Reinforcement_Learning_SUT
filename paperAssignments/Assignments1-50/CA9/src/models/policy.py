from typing import Tuple

import torch
import torch.nn as nn
import torch.distributions as D


class GaussianPolicy(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, a_dim)
        self.logstd = nn.Parameter(torch.zeros(a_dim))

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(s)
        mu = self.mu_head(h)
        std = self.logstd.exp() + 1e-6
        return mu, std

    def sample(self, s: torch.Tensor) -> torch.Tensor:
        mu, std = self.forward(s)
        dist = D.Normal(mu, std)
        return dist.rsample()

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mu, std = self.forward(s)
        dist = D.Normal(mu, std)
        return dist.log_prob(a).sum(-1)


