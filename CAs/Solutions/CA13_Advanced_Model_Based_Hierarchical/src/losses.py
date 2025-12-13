import torch
import torch.nn.functional as F
import torch.distributions as dist
from src.config import WorldModelConfig, ManagerConfig, WorkerConfig

def world_model_loss(
    config: WorldModelConfig,
    prior_dist: dist.Distribution,
    posterior_dist: dist.Distribution,
    reconstructed_observations: dist.Distribution,
    predicted_rewards: dist.Distribution,
    true_observations: torch.Tensor,
    true_rewards: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Computes the World Model loss as described in the README.
    Combines reconstruction losses for observations and rewards, and KL divergence for latent states.

    Args:
        config (WorldModelConfig): Configuration object for the world model.
        prior_dist (dist.Distribution): Distribution of the prior latent states.
        posterior_dist (dist.Distribution): Distribution of the posterior latent states.
        reconstructed_observations (dist.Distribution): Reconstructed observation distributions.
        predicted_rewards (dist.Distribution): Predicted reward distributions.
        true_observations (torch.Tensor): True observations from the environment.
        true_rewards (torch.Tensor): True rewards from the environment.

    Returns:
        tuple[torch.Tensor, dict]: Total world model loss and a dictionary of individual loss components.
    """
    # 1. Reconstruction Loss for Observations
    # Assuming `reconstructed_observations` is a Normal distribution
    obs_loss = -reconstructed_observations.log_prob(true_observations).mean()

    # 2. Reconstruction Loss for Rewards
    # Assuming `predicted_rewards` is a Normal distribution
    reward_loss = -predicted_rewards.log_prob(true_rewards.unsqueeze(-1)).mean()

    # 3. KL Divergence for Latent States
    # KL(posterior || prior)
    kl_loss = dist.kl_divergence(posterior_dist, prior_dist).mean()
    kl_loss = torch.max(kl_loss, torch.tensor(config.free_nats).to(kl_loss.device))

    # Total World Model Loss
    total_loss = obs_loss + reward_loss + config.kl_loss_scale * kl_loss

    losses_dict = {
        'obs_loss': obs_loss.item(),
        'reward_loss': reward_loss.item(),
        'kl_loss': kl_loss.item(),
        'total_wm_loss': total_loss.item()
    }

    return total_loss, losses_dict

def manager_loss(
    config: ManagerConfig,
    manager_actor_output: torch.Tensor, # This would be the goal predicted by the manager
    current_latent_states: torch.Tensor, # S_t
    achieved_latent_states: torch.Tensor, # S_{t+N}
    manager_critic_values: torch.Tensor,
    target_manager_critic_values: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Computes the Manager's loss. Manager aims to set goals that maximize intrinsic rewards.
    Uses a policy gradient approach combined with value estimation.

    Args:
        config (ManagerConfig): Configuration for the manager.
        manager_actor_output (torch.Tensor): The goal vector predicted by the manager actor.
        current_latent_states (torch.Tensor): The latent state at the beginning of the manager's decision horizon.
        achieved_latent_states (torch.Tensor): The latent state achieved by the worker after N steps.
        manager_critic_values (torch.Tensor): Current value estimates from the manager critic.
        target_manager_critic_values (torch.Tensor): Target value estimates for the manager critic.

    Returns:
        tuple[torch.Tensor, dict]: Total manager loss and a dictionary of individual loss components.
    """

    # Intrinsic Reward for Manager: negative distance between achieved state and goal
    # Assuming manager_actor_output is the goal (g_t)
    # And achieved_latent_states is s_{t+N}
    intrinsic_reward = -F.mse_loss(achieved_latent_states, manager_actor_output, reduction='none').mean(dim=-1, keepdim=True)

    # Actor Loss (Policy Gradient)
    # Here we would need an actual policy distribution for the actor. For now, we will use a placeholder.
    # For a deterministic actor, we would maximize the Q-value for the chosen action (goal).
    # For simplicity, we can use the intrinsic reward directly as a signal for the actor.
    actor_loss = -intrinsic_reward.mean() # Simple: manager wants to set goals that lead to high intrinsic reward

    # Critic Loss (Value Function Learning)
    critic_loss = F.mse_loss(manager_critic_values, target_manager_critic_values)

    # Total Manager Loss
    total_loss = actor_loss + critic_loss

    losses_dict = {
        'manager_actor_loss': actor_loss.item(),
        'manager_critic_loss': critic_loss.item(),
        'manager_total_loss': total_loss.item(),
        'manager_intrinsic_reward': intrinsic_reward.mean().item()
    }

    return total_loss, losses_dict

def worker_loss(
    config: WorkerConfig,
    worker_action_dist: dist.Distribution,
    true_actions: torch.Tensor,
    current_latent_states: torch.Tensor, # s_t
    manager_goal: torch.Tensor, # g_t
    achieved_latent_states: torch.Tensor, # s_{t+k}
    extrinsic_rewards: torch.Tensor,
    worker_critic_values: torch.Tensor,
    target_worker_critic_values: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Computes the Worker's loss. Worker aims to execute actions to achieve the manager's goal
    while also considering extrinsic rewards.

    Args:
        config (WorkerConfig): Configuration for the worker.
        worker_action_dist (dist.Distribution): Distribution of actions predicted by the worker actor.
        true_actions (torch.Tensor): Actions taken in the environment (for supervised learning if applicable, or for log_prob).
        current_latent_states (torch.Tensor): Latent state at time t.
        manager_goal (torch.Tensor): Goal set by the manager.
        achieved_latent_states (torch.Tensor): Latent state at time t+k after worker action.
        extrinsic_rewards (torch.Tensor): Extrinsic rewards from the environment.
        worker_critic_values (torch.Tensor): Current value estimates from the worker critic.
        target_worker_critic_values (torch.Tensor): Target value estimates for the worker critic.

    Returns:
        tuple[torch.Tensor, dict]: Total worker loss and a dictionary of individual loss components.
    """

    # Intrinsic Reward for Worker: negative distance from current state to goal
    intrinsic_reward = -F.mse_loss(achieved_latent_states, manager_goal.unsqueeze(1), reduction='none').mean(dim=-1, keepdim=True)

    # Combined Reward for Worker
    # The extrinsic_rewards should be shaped (B, N, 1) and intrinsic_reward (B, N, 1)
    combined_reward = (config.intrinsic_reward_weight * intrinsic_reward) + (config.extrinsic_reward_weight * extrinsic_rewards)

    # Actor Loss (Policy Gradient)
    # Assuming true_actions are actions taken by the worker in imagined trajectory
    actor_loss = (-worker_action_dist.log_prob(true_actions.squeeze(-1)) * combined_reward.detach().squeeze(-1)).mean()

    # Critic Loss (Value Function Learning)
    critic_loss = F.mse_loss(worker_critic_values, target_worker_critic_values)

    # Total Worker Loss
    total_loss = actor_loss + critic_loss

    losses_dict = {
        'worker_actor_loss': actor_loss.item(),
        'worker_critic_loss': critic_loss.item(),
        'worker_total_loss': total_loss.item(),
        'worker_intrinsic_reward': intrinsic_reward.mean().item(),
        'worker_combined_reward': combined_reward.mean().item()
    }

    return total_loss, losses_dict


