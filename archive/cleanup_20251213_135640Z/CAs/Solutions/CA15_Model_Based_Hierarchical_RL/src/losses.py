import torch
import torch.nn.functional as F


def dynamics_model_loss(predicted_next_states: torch.Tensor, actual_next_states: torch.Tensor,
                        predicted_rewards: torch.Tensor, actual_rewards: torch.Tensor) -> torch.Tensor:
    """Calculates the combined loss for the dynamics model (state and reward prediction)."""
    loss_state = F.mse_loss(predicted_next_states, actual_next_states)
    loss_reward = F.mse_loss(predicted_rewards, actual_rewards)
    return loss_state + loss_reward


def q_function_loss(q_values: torch.Tensor, target_q_values: torch.Tensor) -> torch.Tensor:
    """Calculates the Mean Squared Error loss for Q-function updates."""
    return F.mse_loss(q_values, target_q_values)


def intrinsic_reward_loss(current_state: torch.Tensor, next_state: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
    """Calculates the intrinsic reward for the worker based on progress towards a subgoal."""
    # Negative Euclidean distance reduction to the subgoal
    # Smaller distance to subgoal is better, so we want to maximize the reduction
    distance_before = torch.norm(current_state - subgoal, dim=-1)
    distance_after = torch.norm(next_state - subgoal, dim=-1)
    reward = distance_before - distance_after
    return reward


def policy_gradient_loss(log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """Calculates the policy gradient loss."""
    return -(log_probs * advantages).mean()


def actor_critic_loss(
    log_probs: torch.Tensor, values: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates actor and critic losses for actor-critic methods."""
    # Calculate target values for critic
    with torch.no_grad():
        targets = rewards + gamma * values * (1 - dones)

    # Critic loss (MSE between predicted values and targets)
    critic_loss = F.mse_loss(values, targets)

    # Advantages for actor
    advantages = (targets - values).detach()
    actor_loss = -(log_probs * advantages).mean()

    return actor_loss, critic_loss


