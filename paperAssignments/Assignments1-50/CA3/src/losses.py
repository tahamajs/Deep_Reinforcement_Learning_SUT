from typing import Optional
import torch


def reinforce_loss(
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the REINFORCE policy gradient loss.

    Args:
        log_probs: tensor of shape (T,) or (batch,) containing log probability
                   of each action taken.
        returns: tensor of same shape containing the discounted returns (G_t).
        baseline: optional tensor of same shape containing baseline estimates
                  (e.g., value function). If provided, `returns - baseline` is used.

    Returns:
        scalar loss tensor (to minimize) equal to -mean(log_prob * advantage).
    """
    if baseline is not None:
        advantage = returns - baseline
    else:
        advantage = returns
    # Ensure shapes match
    if log_probs.shape != advantage.shape:
        log_probs = log_probs.reshape(advantage.shape)
    loss = -(log_probs * advantage.detach()).mean()
    return loss


def entropy_loss_from_logits(logits: torch.Tensor, coeff: float = 0.0) -> torch.Tensor:
    """Return an entropy regularization term from logits.

    This returns -coeff * entropy so it can be added directly to the optimization
    objective when minimizing (i.e., encourage exploration when coeff>0).
    """
    if coeff == 0.0:
        return torch.tensor(0.0, device=logits.device)
    probs = torch.softmax(logits, dim=-1)
    logp = torch.log(probs + 1e-8)
    entropy = -(probs * logp).sum(dim=-1).mean()
    return -coeff * entropy












