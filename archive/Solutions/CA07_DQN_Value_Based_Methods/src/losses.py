import torch
import torch.nn.functional as F

def c51_loss(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """
    Computes the Categorical DQN (C51) loss, which is the negative log-likelihood
    of the target distribution under the predicted distribution.

    Args:
        logits: The predicted logits from the CategoricalQNetwork, shape (batch_size, action_dim, n_atoms).
        target_probs: The projected target probability distribution, shape (batch_size, action_dim, n_atoms).

    Returns:
        The scalar C51 loss.
    """
    # Compute log probabilities from logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Compute KL-divergence, which for a categorical distribution is equivalent
    # to cross-entropy if target_probs are one-hot, or negative log-likelihood if target_probs are general distributions.
    # We sum over atoms and then over actions, finally mean over batch.
    loss = -torch.sum(target_probs * log_probs, dim=-1) # Sum over atoms for each action
    loss = torch.sum(loss, dim=-1) # Sum over actions
    return loss.mean() # Mean over batch


