from __future__ import annotations
from typing import Optional
import torch


def policy_loss_from_logprob(
    log_probs: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    """
    Standard policy gradient loss (to minimize).
    Args:
        log_probs: (B,)
        advantages: (B,)
    Returns:
        scalar loss
    """
    return -(log_probs * advantages).mean()


def lagrangian_loss(
    policy_loss: torch.Tensor,
    constraint_value: torch.Tensor,
    multiplier_value: float,
    constraint_threshold: float,
) -> torch.Tensor:
    """
    L = policy_loss + mu * (constraint - c)
    Note: policy_loss should already be a scalar.
    """
    penalty = multiplier_value * (constraint_value - constraint_threshold)
    return policy_loss + penalty


def compute_constraint(batch_constraints: torch.Tensor) -> torch.Tensor:
    """
    Aggregates constraint values from a batch into a scalar metric.
    E.g., mean constraint violation across episodes/samples.
    """
    return batch_constraints.mean()








