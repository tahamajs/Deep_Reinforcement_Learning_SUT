import torch
import torch.nn.functional as F
from typing import Tuple

from src.config import config

def decision_transformer_loss(
    state_preds: torch.Tensor,
    action_preds: torch.Tensor,
    return_preds: torch.Tensor,
    state_targets: torch.Tensor,
    action_targets: torch.Tensor,
    return_targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the combined loss for the Decision Transformer, including MSE for states and returns,
    and MSE for actions (or Cross-Entropy for discrete actions).
    """
    # Reshape targets to match predictions
    action_targets = action_targets.reshape(-1, action_preds.shape[-1])
    state_targets = state_targets.reshape(-1, state_preds.shape[-1])
    return_targets = return_targets.reshape(-1, return_preds.shape[-1])

    action_preds = action_preds.reshape(-1, action_preds.shape[-1])
    state_preds = state_preds.reshape(-1, state_preds.shape[-1])
    return_preds = return_preds.reshape(-1, return_preds.shape[-1])

    # Action loss (MSE for continuous, CrossEntropy for discrete - assuming one-hot for simplicity here)
    action_loss = F.mse_loss(action_preds, action_targets)

    # State and Return losses
    state_loss = F.mse_loss(state_preds, state_targets)
    return_loss = F.mse_loss(return_preds, return_targets)

    # Weighted total loss
    total_loss = action_loss + config.STATE_LOSS_WEIGHT * state_loss + config.RETURN_LOSS_WEIGHT * return_loss
    
    return total_loss, action_loss, state_loss, return_loss

def ewc_loss(
    model: torch.nn.Module,
    fisher_information: dict,
    old_params: dict,
    lambda_ewc: float = config.EWC_LAMBDA
) -> torch.Tensor:
    """
    Computes the Elastic Weight Consolidation (EWC) regularization loss.
    """
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher_information:
            loss += (fisher_information[name] * (param - old_params[name])**2).sum()
    return lambda_ewc * loss

def logic_regularization_loss(
    predicted_actions: torch.Tensor,
    symbolic_constraints: torch.Tensor,
    gamma: float = config.SYMBOLIC_WEIGHT
) -> torch.Tensor:
    """
    Computes a logic regularization loss based on symbolic constraints.
    This is a placeholder for a more complex interaction. Here, we assume a direct penalty.
    """
    # Example: Penalize actions that violate a binary symbolic constraint
    # Assuming symbolic_constraints is 0 for allowed actions and 1 for disallowed.
    # And predicted_actions are probabilities or logits for each action.
    
    # For simplicity, if predicted_actions is logits, and constraints indicate which actions are forbidden
    # We want to minimize probabilities of forbidden actions.
    if predicted_actions.shape == symbolic_constraints.shape:
        # Assuming symbolic_constraints is a one-hot like tensor where 1 indicates a forbidden action
        # We want to push down the probability of these actions.
        # This is a simplified example; actual logic would depend on the symbolic system.
        violation_penalty = (predicted_actions * symbolic_constraints).sum(dim=-1).mean()
    else:
        # Placeholder for more complex interaction or if constraints are scalar per state
        violation_penalty = torch.zeros(1, device=predicted_actions.device)

    return gamma * violation_penalty


