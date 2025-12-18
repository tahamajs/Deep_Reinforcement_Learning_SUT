from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import CFG


class QRDQNNetwork(nn.Module):
    """
    Quantile Regression DQN: outputs quantile values per action.
    Output shape: [B, action_dim, num_quantiles]
    """
    def __init__(self, state_dim: int, action_dim: int, num_quantiles: int = CFG.qr_num_quantiles, hidden: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim * num_quantiles)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = out.view(-1, self.action_dim, self.num_quantiles)
        return out


def quantile_huber_loss(predictions: torch.Tensor, targets: torch.Tensor, taus: torch.Tensor, kappa: float = CFG.qr_kappa) -> torch.Tensor:
    """
    Quantile Huber loss between predicted quantiles and target quantiles.
    predictions: [B, N] where N = num_quantiles (flattened per-action usage)
    targets: [B, M] target quantiles
    taus: [N] quantile fractions for predictions
    Returns scalar loss.
    """
    # pairwise differences
    # predictions: [B, N], targets: [B, M]
    diff = targets.unsqueeze(1) - predictions.unsqueeze(2)  # [B, N, M]
    abs_diff = diff.abs()

    huber = torch.where(abs_diff <= kappa, 0.5 * diff.pow(2), kappa * (abs_diff - 0.5 * kappa))
    # taus: [N], need shape [1, N, 1]
    weight = (taus.unsqueeze(0).unsqueeze(2) - (diff < 0).float()).abs()
    loss = (weight * huber).mean()
    return loss




