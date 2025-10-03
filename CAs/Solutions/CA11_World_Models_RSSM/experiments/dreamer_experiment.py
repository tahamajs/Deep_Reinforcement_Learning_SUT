"""
Dreamer Experiment Script

This module provides a complete experiment script for training and evaluating
Dreamer agents with world models and latent space planning.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import argparse
import json
from pathlib import Path

from models.vae import VariationalAutoencoder
from models.dynamics import LatentDynamicsModel
from models.reward_model import RewardModel
from models.world_model import WorldModel
from agents.dreamer_agent import DreamerAgent

from environments.continuous_cartpole import ContinuousCartPole
from environments.continuous_pendulum import ContinuousPendulum

from utils.data_collection import collect_world_model_data, collect_rollout_data
from utils.visualization import plot_dreamer_training, plot_world_model_training


def run_dreamer_experiment(
    config: Dict[str, Any],
) -> tuple[DreamerAgent, Dict[str, Any]]:
    """
    Run a complete Dreamer experiment.

    Args:
        config: Experiment configuration dictionary

    Returns:
        Tuple of (trained_dreamer_agent, results)
    """
    # Set random seeds
    np.random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    if config["env_name"] == "continuous_cartpole":
        env = ContinuousCartPole()
    elif config["env_name"] == "continuous_pendulum":
        env = ContinuousPendulum()
    else:
        raise ValueError(f"Unknown environment: {config['env_name']}")

    print(f"Environment: {env.name}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Collect initial data for world model training
    print("Collecting initial data for world model training...")
    world_model_data = collect_world_model_data(
        env=env,
        steps=config["world_model_data_steps"],
        episodes=config.get("world_model_data_episodes"),
        seed=config.get("seed", 42),
    )

    print(
        f"Collected {len(world_model_data['observations'])} transitions for world model"
    )

    # Create world model components
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_dim = config["latent_dim"]

    vae = VariationalAutoencoder(
        obs_dim=obs_dim, latent_dim=latent_dim, hidden_dims=config["vae_hidden_dims"]
    ).to(device)

    dynamics = LatentDynamicsModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dims=config["dynamics_hidden_dims"],
        stochastic=config.get("stochastic_dynamics", True),
    ).to(device)

    reward_model = RewardModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dims=config["reward_hidden_dims"],
    ).to(device)

    world_model = WorldModel(vae, dynamics, reward_model).to(device)

    print(f"World model created:")
    print(f"- VAE: {obs_dim} -> {latent_dim}")
    print(f"- Dynamics: {latent_dim} + {action_dim} -> {latent_dim}")
    print(f"- Reward: {latent_dim} + {action_dim} -> 1")

    # Train world model
    print("Training world model...")
    world_model_optimizer = torch.optim.Adam(
        world_model.parameters(), lr=config["world_model_lr"]
    )

    world_model_losses = []
    for epoch in range(config["world_model_epochs"]):
        # Sample batch
        batch_size = config["world_model_batch_size"]
        indices = np.random.choice(
            len(world_model_data["observations"]), batch_size, replace=False
        )

        batch = {
            "observations": torch.FloatTensor(
                world_model_data["observations"][indices]
            ).to(device),
            "actions": torch.FloatTensor(world_model_data["actions"][indices]).to(
                device
            ),
            "next_observations": torch.FloatTensor(
                world_model_data["next_observations"][indices]
            ).to(device),
            "rewards": torch.FloatTensor(world_model_data["rewards"][indices]).to(
                device
            ),
        }

        # Compute loss
        world_model_optimizer.zero_grad()
        losses = world_model.compute_loss(
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["rewards"],
            beta=config.get("beta_value", 1.0),
        )

        # Backward pass
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
        world_model_optimizer.step()

        world_model_losses.append({k: v.item() for k, v in losses.items()})

        if (epoch + 1) % config.get("world_model_print_interval", 50) == 0:
            print(
                f"World Model Epoch {epoch+1}: Loss = {losses['total_loss'].item():.4f}"
            )

    print("World model training completed!")

    # Create Dreamer agent
    dreamer = DreamerAgent(
        world_model=world_model,
        state_dim=latent_dim,
        action_dim=action_dim,
        device=device,
        actor_lr=config["actor_lr"],
        critic_lr=config["critic_lr"],
        gamma=config.get("gamma", 0.99),
        lambda_=config.get("lambda_", 0.95),
        imagination_horizon=config.get("imagination_horizon", 15),
    )

    print(f"Dreamer agent created:")
    print(f"- Imagination horizon: {dreamer.imagination_horizon}")
    print(f"- Discount factor: {dreamer.gamma}")
    print(f"- Actor learning rate: {dreamer.actor_lr}")
    print(f"- Critic learning rate: {dreamer.critic_lr}")

    # Training loop
    print("Starting Dreamer training...")
    episode_rewards = []
    episode_lengths = []

    for episode in range(config["dreamer_episodes"]):
        # Collect rollout data
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(config.get("max_steps", 200)):
            # Select action using Dreamer agent
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            latent_state = world_model.encode_observations(obs_tensor).squeeze(0)

            # Use actor to select action
            action, _ = dreamer.actor.sample(latent_state.unsqueeze(0))
            action = action.squeeze(0).cpu().numpy()

            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            dreamer.store_transition(obs, action, reward, next_obs, done)

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Update world model
        if len(dreamer.buffer) >= config.get("world_model_update_frequency", 10):
            world_model_losses_episode = dreamer.update_world_model(
                batch_size=config.get("world_model_batch_size", 32)
            )

        # Update actor-critic
        if len(dreamer.buffer) >= config.get("actor_critic_update_frequency", 5):
            actor_critic_losses = dreamer.update_actor_critic(
                batch_size=config.get("actor_critic_batch_size", 32)
            )

        # Print progress
        if (episode + 1) % config.get("print_interval", 10) == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(
                f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.1f}"
            )

    print("Dreamer training completed!")

    # Final evaluation
    print("Evaluating final performance...")
    eval_rewards = []
    eval_lengths = []

    for _ in range(config.get("eval_episodes", 10)):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(config.get("max_steps", 200)):
            # Select action using trained agent
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            latent_state = world_model.encode_observations(obs_tensor).squeeze(0)

            # Use deterministic action selection
            action = dreamer.actor.get_action(
                latent_state.unsqueeze(0), deterministic=True
            )
            action = action.squeeze(0).cpu().numpy()

            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            if done:
                break

        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)

    final_avg_reward = np.mean(eval_rewards)
    final_avg_length = np.mean(eval_lengths)

    print(
        f"Final evaluation: Avg Reward = {final_avg_reward:.2f}, Avg Length = {final_avg_length:.1f}"
    )

    # Prepare results
    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "eval_rewards": eval_rewards,
        "eval_lengths": eval_lengths,
        "final_avg_reward": final_avg_reward,
        "final_avg_length": final_avg_length,
        "world_model_losses": world_model_losses,
    }

    # Generate plots
    if config.get("save_plots", True):
        save_dir = Path(config.get("save_dir", "results"))
        save_dir.mkdir(exist_ok=True)

        # Dreamer training progress
        plot_dreamer_training(
            dreamer,
            title=f"Dreamer Training - {env.name}",
            save_path=save_dir / "dreamer_training_progress.png",
        )

        # Episode rewards
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, "b-", linewidth=2)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(episode_lengths, "r-", linewidth=2)
        plt.title("Episode Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            save_dir / "dreamer_episode_metrics.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Save model
    if config.get("save_model", True):
        save_dir = Path(config.get("save_dir", "results"))
        save_dir.mkdir(exist_ok=True)

        torch.save(
            {
                "dreamer_state_dict": dreamer.state_dict(),
                "world_model_state_dict": world_model.state_dict(),
                "config": config,
                "results": results,
            },
            save_dir / "dreamer_model.pth",
        )

    return dreamer, results


def main():
    """Main function for running Dreamer experiments."""
    parser = argparse.ArgumentParser(description="Dreamer Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dreamer_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="continuous_cartpole",
        choices=["continuous_cartpole", "continuous_pendulum"],
        help="Environment to use",
    )
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument(
        "--dreamer_episodes",
        type=int,
        default=200,
        help="Number of Dreamer training episodes",
    )
    parser.add_argument(
        "--world_model_epochs",
        type=int,
        default=100,
        help="Number of world model training epochs",
    )
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Directory to save results"
    )

    args = parser.parse_args()

    # Default configuration
    config = {
        "env_name": args.env,
        "latent_dim": args.latent_dim,
        "vae_hidden_dims": [256, 128],
        "dynamics_hidden_dims": [256, 128],
        "reward_hidden_dims": [128, 64],
        "stochastic_dynamics": True,
        "world_model_lr": 1e-3,
        "world_model_epochs": args.world_model_epochs,
        "world_model_batch_size": 64,
        "world_model_data_steps": 5000,
        "world_model_data_episodes": 20,
        "world_model_print_interval": 50,
        "actor_lr": 8e-5,
        "critic_lr": 8e-5,
        "gamma": 0.99,
        "lambda_": 0.95,
        "imagination_horizon": 15,
        "dreamer_episodes": args.dreamer_episodes,
        "max_steps": 200,
        "world_model_update_frequency": 10,
        "actor_critic_update_frequency": 5,
        "world_model_batch_size": 32,
        "actor_critic_batch_size": 32,
        "print_interval": 10,
        "eval_episodes": 10,
        "save_plots": True,
        "save_model": True,
        "save_dir": args.save_dir,
        "seed": 42,
        "beta_value": 1.0,
    }

    # Load configuration file if provided
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            file_config = json.load(f)
            config.update(file_config)

    # Run experiment
    dreamer, results = run_dreamer_experiment(config)

    print("Experiment completed successfully!")
    print(f"Final average reward: {results['final_avg_reward']:.2f}")
    print(f"Final average episode length: {results['final_avg_length']:.1f}")


if __name__ == "__main__":
    main()
