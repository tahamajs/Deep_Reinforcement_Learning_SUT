import torch
import torch.nn.functional as F
from typing import Tuple, Dict

def critic_loss_fn(
    current_q_values: torch.Tensor,
    target_q_values: torch.Tensor
) -> torch.Tensor:
    """Calculates the Mean Squared Error (MSE) loss for the critic network.

    Args:
        current_q_values (torch.Tensor): The Q-values predicted by the current critic network.
        target_q_values (torch.Tensor): The target Q-values (from target network and rewards).

    Returns:
        torch.Tensor: The MSE loss.
    """
    return F.mse_loss(current_q_values, target_q_values)

def actor_loss_fn(
    log_probs: torch.Tensor,
    advantages: torch.Tensor
) -> torch.Tensor:
    """Calculates the actor loss using policy gradient theorem.

    Args:
        log_probs (torch.Tensor): Log probabilities of actions taken by the actor.
        advantages (torch.Tensor): Advantage estimates for the actions.

    Returns:
        torch.Tensor: The actor loss (negative of expected advantage).
    """
    return -(log_probs * advantages.detach()).mean()

def maml_policy_loss(
    adapted_policy_output: torch.Tensor,
    target_policy_output: torch.Tensor,
    rewards: torch.Tensor
) -> torch.Tensor:
    """Placeholder for MAML policy loss. Can be cross-entropy for discrete or MSE for continuous.
    In a full MAML MARL setting, this would involve policy gradients and advantages.
    For simplicity, assume a supervised-like loss for policy output for inner loop.
    """
    # This is a simplified placeholder. In a real MAML MARL setup, this would be
    # a policy gradient loss (e.g., -log_prob * advantage) for the inner loop.
    # For demonstration, we use a simple MSE if adapted_policy_output is Q-values or similar.
    # For discrete action space, it could be F.cross_entropy if target_policy_output is actions.
    # For now, a generic loss that assumes 'rewards' as a proxy for desired outcome for simplicity.
    return F.mse_loss(adapted_policy_output, rewards.detach().squeeze(-1))

def maml_communication_loss(
    predicted_messages: torch.Tensor,
    target_messages: torch.Tensor, # Or some other communication metric
    coordination_rewards: torch.Tensor
) -> torch.Tensor:
    """Placeholder for MAML communication loss. Aims to make messages contribute to coordination.
    Could be based on reconstruction, mutual information, or direct impact on rewards.
    """
    # This is a simplified placeholder. A real communication loss might involve
    # mutual information, or a loss that guides messages to be informative.
    # For demonstration, a simple MSE or a loss tied to coordination outcomes.
    return F.mse_loss(predicted_messages, target_messages.detach()) + (-coordination_rewards.mean() * 0.1)

def compute_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float
) -> torch.Tensor:
    """Computes basic one-step TD advantage.

    Args:
        rewards (torch.Tensor): Rewards received at current step.
        values (torch.Tensor): Value estimates at current state.
        next_values (torch.Tensor): Value estimates at next state.
        dones (torch.Tensor): Done flags.
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: One-step TD advantage.
    """
    td_target = rewards + gamma * next_values * (1 - dones)
    advantage = td_target - values
    return advantage
