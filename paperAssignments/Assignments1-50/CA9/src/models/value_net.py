from typing import Tuple

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, s_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


def expectile_loss(diff: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """Expectile loss used by IQL-style value updates.
    diff = target - V(s)
    """
    weight = torch.where(diff > 0, tau, (1.0 - tau))
    return (weight * (diff**2)).mean()
