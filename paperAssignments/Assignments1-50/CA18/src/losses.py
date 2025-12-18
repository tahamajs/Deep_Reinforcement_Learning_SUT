from __future__ import annotations
from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float = 0.95,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE).
    rewards, values, dones: shape (T, B) or (B, T) depending on convention; here we expect (B, T)
    """
    # Expect shapes: (B, T)
    if rewards.dim() != 2:
        raise ValueError("rewards must be shape (B, T)")
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(B, device=rewards.device)
    for t in range(T - 1, -1, -1):
        nextnonterminal = 1.0 - dones[:, t]
        nextvalues = (
            values[:, t + 1]
            if t + 1 < values.shape[1]
            else torch.zeros(B, device=values.device)
        )
        delta = rewards[:, t] + gamma * nextvalues * nextnonterminal - values[:, t]
        advantages[:, t] = lastgaelam = (
            delta + gamma * lam * nextnonterminal * lastgaelam
        )
    return advantages


def policy_value_losses(
    logp: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    value_coef: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Compute policy loss and value loss."""
    # policy loss (negative for gradient descent)
    policy_loss = -(logp * advantages.detach()).mean()
    value_loss = F.mse_loss(values, returns)
    total = policy_loss + value_coef * value_loss
    return {"policy_loss": policy_loss, "value_loss": value_loss, "total": total}


def entropy_loss(entropy: torch.Tensor, coef: float = 0.01) -> torch.Tensor:
    return -coef * entropy.mean()


def kl_divergence_from_logits(
    logits_p: torch.Tensor, logits_q: torch.Tensor
) -> torch.Tensor:
    """Compute KL(p || q) for categorical distributions given logits."""
    p = torch.distributions.Categorical(logits=logits_p)
    q = torch.distributions.Categorical(logits=logits_q)
    # use expected log prob difference
    pk = torch.distributions.kl_divergence(p, q)
    return pk.mean()














