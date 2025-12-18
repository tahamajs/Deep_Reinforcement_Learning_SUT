"""
Utility helpers for CA8.
"""

from typing import Iterable, Optional

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Soft-update parameters: target <- tau*source + (1-tau)*target"""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau)
        tp.data.add_(tau * sp.data)


def cvar(particles: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    Compute CVaR (lower tail) estimator from particles.
    particles: [B, N] or [B, N, 1]
    returns: [B, 1]
    """
    if particles.dim() == 3:
        particles = particles.squeeze(-1)
    sorted_p, _ = torch.sort(particles, dim=1)
    k = max(1, int(alpha * sorted_p.size(1)))
    return sorted_p[:, :k].mean(dim=1, keepdim=True)


def mlp(
    output_size: int,
    hidden: Iterable[int] = (256, 256),
    input_size: Optional[int] = None,
):
    """
    Build a simple MLP. If input_size is None, the first Linear must be provided externally.
    Returns nn.Sequential.
    """
    layers = []
    if input_size is not None:
        in_ch = input_size
    else:
        in_ch = None
    for h in hidden:
        if in_ch is None:
            # placeholder linear that should be replaced by user
            layers.append(torch.nn.Linear(1, h))  # pragma: no cover (placeholder)
            in_ch = h
        else:
            layers.append(torch.nn.Linear(in_ch, h))
            layers.append(torch.nn.ReLU())
            in_ch = h
    layers.append(torch.nn.Linear(in_ch, output_size))
    return torch.nn.Sequential(*layers)









