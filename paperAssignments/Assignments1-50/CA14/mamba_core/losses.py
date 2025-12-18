"""Loss utilities for MAMBA-PEAC simplified implementation.
"""
from typing import Tuple, List

import torch
import torch.nn.functional as F


def kl_normal(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor = None, logvar_p: torch.Tensor = None) -> torch.Tensor:
    """KL between two diagonal Gaussians. Returns per-sample sum over latent dims.

    If prior is standard normal, mu_p/logvar_p can be None.
    """
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * ( (logvar_p - logvar_q) + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0 )
    return kl.sum(-1)


def td_lambda(rs: List[torch.Tensor], gammas: List[torch.Tensor], values: List[torch.Tensor], lambda_: float = 0.95) -> List[torch.Tensor]:
    """Compute TD(lambda) targets for an imagined rollout.

    rs, gammas, values are lists/tuples of length H (each tensor shaped (B,)).
    Returns list of tensors targets of length H.
    """
    G = values[-1]
    targets: List[torch.Tensor] = []
    for r, g, v in reversed(list(zip(rs, gammas, values))):
        G = r + g * ((1 - lambda_) * v + lambda_ * G)
        targets.append(G)
    targets.reverse()
    return targets


def world_model_loss(recon_x: torch.Tensor, x: torch.Tensor, reward_pred: torch.Tensor, reward: torch.Tensor, kl_z: torch.Tensor, kl_m: torch.Tensor, beta_z: float = 1.0, beta_m: float = 0.5, free_bits_z: float = 1.0, free_bits_m: float = 1.0) -> torch.Tensor:
    """Compute a simplified world model loss combining reconstruction, reward and KLs.

    Args:
        recon_x: reconstructed observations (B, T, D) or (B, D)
        x: target observations
        reward_pred: predicted rewards
        reward: target rewards
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='none')
    recon_loss = recon_loss.mean()
    reward_loss = F.mse_loss(reward_pred, reward, reduction='mean')
    kl_z_clamped = torch.clamp(kl_z - free_bits_z, min=0.0).mean()
    kl_m_clamped = torch.clamp(kl_m - free_bits_m, min=0.0).mean()
    loss = recon_loss + reward_loss + beta_z * kl_z_clamped + beta_m * kl_m_clamped
    return loss
