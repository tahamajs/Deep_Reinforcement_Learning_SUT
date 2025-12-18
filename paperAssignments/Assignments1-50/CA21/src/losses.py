from typing import Optional
import torch


def policy_gradient_loss(
    log_probs: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    """
    Compute the (negative) policy gradient loss for a batch.

    Args:
        log_probs: (batch,) log probability of taken actions
        advantages: (batch,) advantage estimates
    Returns:
        scalar loss tensor
    """
    if log_probs.shape != advantages.shape:
        raise ValueError("log_probs and advantages must have same shape")
    return -(log_probs * advantages).mean()


def value_mse_loss(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Simple mean-squared-error loss for value function regression.
    """
    if values.shape != targets.shape:
        raise ValueError("values and targets must have same shape")
    return torch.mean((values - targets) ** 2)

