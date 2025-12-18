from typing import Optional
import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared error for value regression."""
    return torch.mean((pred - target) ** 2)


def policy_gradient_loss(
    logp: torch.Tensor, advantage: torch.Tensor, reduction: Optional[str] = "mean"
) -> torch.Tensor:
    """
    Simple policy gradient (REINFORCE) loss: -logp * advantage
    logp: shape (batch,)
    advantage: shape (batch,)
    """
    loss = -(logp * advantage)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss















