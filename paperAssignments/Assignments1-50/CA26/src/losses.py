from __future__ import annotations

import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error with shape checks.

    Returns a scalar tensor.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have same shape: {pred.shape} vs {target.shape}")
    return torch.mean((pred - target) ** 2)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss (smooth L1) implemented explicitly for clarity."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=abs_diff.device, dtype=abs_diff.dtype))
    linear = abs_diff - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta * linear)
