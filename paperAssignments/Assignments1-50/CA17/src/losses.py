from __future__ import annotations

import torch
from torch import Tensor


def policy_gradient_loss(log_probs: Tensor, advantages: Tensor) -> Tensor:
    """Compute the standard policy gradient loss (to minimize).

    Args:
        log_probs: tensor of log probabilities of selected actions, shape (N,)
        advantages: tensor of advantages (or returns - baseline), shape (N,)
    Returns:
        scalar loss
    """
    if log_probs.shape != advantages.shape:
        advantages = advantages.reshape(log_probs.shape)
    return -(log_probs * advantages).mean()


def entropy_loss(logits: Tensor, coeff: float = 0.01) -> Tensor:
    """Entropy regularization loss (to minimize: -entropy * coeff)."""
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    return -coeff * ent














