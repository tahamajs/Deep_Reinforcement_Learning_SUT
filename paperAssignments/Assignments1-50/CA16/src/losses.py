from typing import Optional

import torch
import torch.nn.functional as F


def policy_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    entropy_coeff: float = 0.0,
    entropies: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute policy gradient loss (to minimize).

    Args:
        log_probs: (B,) log probabilities of taken actions
        advantages: (B,) advantage estimates
        entropy_coeff: weight for entropy bonus (added as negative to loss)
        entropies: optional (B,) entropy terms for each distribution
    Returns:
        scalar loss tensor
    """
    pg = -(log_probs * advantages).mean()
    ent_bonus = 0.0
    if entropy_coeff != 0.0 and entropies is not None:
        ent_bonus = -entropy_coeff * entropies.mean()
    return pg + ent_bonus


def value_loss(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error value loss.

    Args:
        values: (B,) value predictions
        targets: (B,) bootstrap targets
    """
    return F.mse_loss(values, targets)










