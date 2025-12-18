from typing import Tuple, Optional

import torch


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns.

    Args:
        rewards: Tensor[T,] rewards per step
        values: Tensor[T+1,] value estimates (bootstrap last value)
        dones: Tensor[T,] binary done flags (1.0 if done at step, else 0.0)
        gamma: discount factor (scalar)
        lam: GAE lambda

    Returns:
        advantages: Tensor[T,]
        returns: Tensor[T,] = advantages + values[:-1]
    """
    T = rewards.shape[0]
    device = rewards.device
    adv = torch.zeros(T, dtype=values.dtype, device=device)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * mask * next_value - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    returns = adv + values[:-1]
    return adv, returns


def update_gamma(
    gamma: float,
    varA: float,
    sigma_target: float,
    alpha: float,
    gmin: float,
    gmax: float,
) -> float:
    """
    Simple proportional controller for gamma.
    This function treats gamma as a scalar hyperparameter (no grad).
    """
    gamma_new = gamma + alpha * (sigma_target - varA)
    gamma_clamped = float(max(min(gamma_new, gmax), gmin))
    return gamma_clamped


def update_gamma_with_ema(
    gamma: float,
    varA: float,
    ema_varA: Optional[float],
    alpha: float,
    beta: float,
    sigma_target: float,
    gmin: float,
    gmax: float,
) -> Tuple[float, float]:
    """
    Update EMA of variance and adapt gamma.
    Returns (gamma_new, ema_varA_new)
    """
    if ema_varA is None:
        ema_varA_new = float(varA)
    else:
        ema_varA_new = float(beta * ema_varA + (1.0 - beta) * float(varA))
    gamma_new = update_gamma(gamma, ema_varA_new, sigma_target, alpha, gmin, gmax)
    return gamma_new, ema_varA_new


