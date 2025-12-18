from typing import Tuple
import torch
import torch.nn.functional as F


def quantile_huber_loss(
    quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Quantile Huber loss between predicted `quantiles` and `target_quantiles`.

    Args:
        quantiles: [B, N] - predicted quantiles
        target_quantiles: [B, N] - target quantiles
        taus: [N] or [1, N] - quantile midpoints (tau_i = (i-0.5)/N)
        kappa: huber threshold
    Returns:
        scalar loss
    """
    # u = y - q  -> shape [B, N, N] for pairwise diffs
    u = target_quantiles.unsqueeze(1) - quantiles.unsqueeze(2)
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
    # taus: [N] -> [1, N] if needed
    if taus.dim() == 1:
        taus = taus.view(1, -1)
    # weight: |tau - I(u<0)|
    weight = (taus.unsqueeze(2) - (u.detach() < 0).float()).abs()
    loss = (weight * huber).sum(dim=2).mean(dim=1).mean()
    return loss


def cvar_tail(quantiles: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    Compute CVaR (mean of lower alpha fraction) from quantile estimates.
    quantiles: [B, N] (unsorted or sorted: we'll sort ascending)
    returns: [B, 1]
    """
    q_sorted, _ = torch.sort(quantiles, dim=1)
    k = max(1, int(alpha * q_sorted.size(1)))
    return q_sorted[:, :k].mean(dim=1, keepdim=True)
