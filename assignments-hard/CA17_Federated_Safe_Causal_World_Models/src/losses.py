import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any

def world_model_loss(
    recon_obs: torch.Tensor,
    obs: torch.Tensor,
    recon_reward: torch.Tensor,
    reward: torch.Tensor,
    recon_cost: torch.Tensor,
    cost: torch.Tensor,
    prior_dist: torch.distributions.Normal,
    posterior_dist: torch.distributions.Normal,
    config: Any,
    causal_regularization_loss: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Calculates the total loss for the World Model.

    Combines reconstruction loss (observation, reward, cost), KL divergence for RSSM,
    and an optional causal regularization loss.
    """
    # Observation reconstruction loss
    obs_loss = F.mse_loss(recon_obs, obs) * config.observation_loss_scale

    # Reward reconstruction loss
    reward_loss = F.mse_loss(recon_reward, reward) * config.reward_loss_scale

    # Cost reconstruction loss
    cost_loss = F.mse_loss(recon_cost, cost) * config.cost_loss_scale # Assuming cost_loss_scale in config

    # KL divergence loss for RSSM
    kl_loss = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean() * config.kl_loss_scale
    # Apply free_nats if configured
    kl_loss = torch.max(kl_loss, torch.tensor(config.free_nats).to(kl_loss.device))

    total_loss = obs_loss + reward_loss + cost_loss + kl_loss
    metrics = {"obs_loss": obs_loss.detach(), "reward_loss": reward_loss.detach(),
               "cost_loss": cost_loss.detach(), "kl_loss": kl_loss.detach()}

    if causal_regularization_loss is not None:
        total_loss += causal_regularization_loss * config.causal_regularization_coeff
        metrics["causal_reg_loss"] = causal_regularization_loss.detach()

    return total_loss, metrics

def policy_loss(action_dist: torch.distributions.Normal, value_pred: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
    """Calculates the policy gradient loss (for actor) and value loss (for critic)."""
    # Policy loss (e.g., A2C or PPO style, simplified for now)
    # For CPO, this would be part of a constrained optimization problem
    advantage = target_value - value_pred # Simple advantage for now
    policy_gradient_loss = -(action_dist.log_prob(action_dist.sample()) * advantage.detach()).mean()

    # Value loss (for critic)
    value_loss = F.mse_loss(value_pred, target_value)

    return policy_gradient_loss, value_loss

def safety_loss(predicted_cost: torch.Tensor, actual_cost: torch.Tensor, safety_threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the loss for the SafetyCritic and the constraint violation term.

    Args:
        predicted_cost (torch.Tensor): The cost predicted by the SafetyCritic.
        actual_cost (torch.Tensor): The actual cost observed or imagined.
        safety_threshold (float): The maximum allowed cumulative cost.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The loss for training the SafetyCritic.
            - The constraint violation term for constrained policy optimization.
    """
    # Safety critic loss (e.g., MSE between predicted and actual costs)
    safety_critic_loss = F.mse_loss(predicted_cost, actual_cost)

    # Constraint violation term: max(0, predicted_cost - safety_threshold)
    # This term would be used in a CPO-like algorithm to penalize exceeding the safety budget.
    constraint_violation = F.relu(predicted_cost - safety_threshold).mean()

    return safety_critic_loss, constraint_violation

