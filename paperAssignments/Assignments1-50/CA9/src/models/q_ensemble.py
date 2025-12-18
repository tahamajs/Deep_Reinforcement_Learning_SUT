import math
from typing import List, Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QEnsemble(nn.Module):
    """An ensemble of independent Q-networks with convenient helpers."""

    def __init__(self, state_dim: int, action_dim: int, num_q: int = 4):
        super().__init__()
        self.num_q = num_q
        self.qs = nn.ModuleList([MLP(state_dim + action_dim) for _ in range(num_q)])

    def forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate ensemble on (s,a).
        Returns: mean [B,1], std [B,1]
        """
        sa = torch.cat([s, a], dim=-1)
        outputs = torch.stack([q(sa).unsqueeze(0) for q in self.qs], dim=0)  # [N, B, 1]
        mean = outputs.mean(0)
        std = outputs.std(0, unbiased=False)
        return mean, std

    def all_q(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return stacked Q values shape [N, B, 1]."""
        sa = torch.cat([s, a], dim=-1)
        outputs = torch.stack([q(sa).unsqueeze(0) for q in self.qs], dim=0)
        return outputs

    @torch.no_grad()
    def soft_update_from(self, other: "QEnsemble", tau: float):
        """Polyak average parameters from other ensemble."""
        for p, q in zip(self.parameters(), other.parameters()):
            p.data.copy_(tau * p.data + (1.0 - tau) * q.data)












