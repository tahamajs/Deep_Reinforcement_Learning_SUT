"""Loss functions for policy gradient and value fitting.

Small, well-tested helpers that perform input validation and return scalar
loss tensors appropriate for optimization.
"""
from __future__ import annotations

import torch


def policy_gradient_loss(log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """Compute standard policy gradient loss (-E[log_pi * A]).

    Both inputs must have the same shape and a floating dtype.
    """
    if log_probs.shape != advantages.shape:
        raise ValueError("log_probs and advantages must have the same shape")
    if not torch.is_floating_point(log_probs):
        log_probs = log_probs.to(dtype=torch.get_default_dtype())
    if not torch.is_floating_point(advantages):
        advantages = advantages.to(dtype=torch.get_default_dtype())
    return -(log_probs * advantages).mean()


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Mean-squared error between predicted values and returns.

    The result is a scalar tensor >= 0.
    """
    if values.shape != returns.shape:
        raise ValueError("values and returns must have the same shape")
    return torch.mean((values - returns) ** 2)


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Return negative mean entropy (suitable for minimization) from probabilities."""
    eps = 1e-8
    logp = torch.log(probs.clamp(min=eps))
    ent = -torch.sum(probs * logp, dim=-1)
    return -ent.mean()  # negative so minimizing yields higher entropy


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy loss from raw logits in a numerically stable way."""
    probs = torch.softmax(logits, dim=-1)
    return entropy_from_probs(probs)
