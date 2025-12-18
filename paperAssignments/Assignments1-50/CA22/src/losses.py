from __future__ import annotations
from typing import Optional
import torch


def policy_gradient_loss(
    log_probs: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    """Standard policy gradient (negative objective to minimize).

    log_probs: (batch,)
    advantages: (batch,)
    returns scalar loss (to minimize)
    """
    return -(log_probs * advantages).mean()


def value_loss(pred_values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Simple MSE value loss."""
    return torch.nn.functional.mse_loss(pred_values, targets)


class LagrangianLoss:
    """Combines reward objective with constraint via Lagrange multiplier.

    This class does not perform optimization steps; it only composes the scalar loss
    given a current multiplier `mu`.
    """

    def __init__(self, mu: float = 0.0, constraint_threshold: float = 0.0):
        self.mu = float(mu)
        self.c = float(constraint_threshold)

    def __call__(
        self,
        pg_loss: torch.Tensor,
        constraint_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return combined loss: pg_loss + mu * (mean(constraint) - c).

        The sign convention: pg_loss is already a loss to minimize (negative reward);
        We add penalty mu * (expected_constraint - c).
        """
        mean_constraint = constraint_values.mean()
        penalty = self.mu * (mean_constraint - self.c)
        # Ensure final is a torch scalar
        return pg_loss + penalty

    def set_mu(self, new_mu: float) -> None:
        self.mu = float(new_mu)
