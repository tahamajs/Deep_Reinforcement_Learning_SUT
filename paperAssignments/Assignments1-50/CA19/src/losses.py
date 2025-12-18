from typing import Tuple
import torch
import torch.nn.functional as F


def value_ensemble_variance(values: torch.Tensor) -> torch.Tensor:
    """Compute variance across ensemble for each batch element.

    values: (M, B)
    returns: (B,) variance
    """
    return values.var(dim=0, unbiased=False)


def critic_loss(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error between ensemble mean and targets.

    values: (M, B)
    targets: (B,)
    """
    mean_v = values.mean(0)
    return F.mse_loss(mean_v, targets)


def actor_loss(logits: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor, value_var: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    """Policy gradient loss with uncertainty penalty.

    logits: (B, A)
    actions: (B,)
    advantages: (B,)
    value_var: (B,)
    beta: weight for uncertainty bonus
    """
    logp = F.log_softmax(logits, dim=-1)
    chosen = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    # augmented advantage: encourage taking actions in states with high uncertainty
    aug_adv = advantages + beta * value_var
    # policy gradient loss (we minimize -E[aug_adv * log pi])
    loss = -(aug_adv.detach() * chosen).mean()
    return loss
