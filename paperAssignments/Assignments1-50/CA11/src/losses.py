import torch
import torch.nn.functional as F
from typing import Tuple


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE reconstruction loss for continuous tokens/pixels.
    Args:
        pred: (..., d)
        target: (..., d)
    """
    return F.mse_loss(pred, target)


def reward_loss(pred_reward: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_reward, reward)


def total_model_loss(
    pred_obs: torch.Tensor,
    obs: torch.Tensor,
    pred_reward: torch.Tensor,
    reward: torch.Tensor,
    weights: Tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
    loss_obs = reconstruction_loss(pred_obs, obs)
    loss_r = reward_loss(pred_reward, reward)
    return weights[0] * loss_obs + weights[1] * loss_r


def vq_reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss for VQ-VAE reconstructions (MSE).
    """
    return F.mse_loss(recon, target)















