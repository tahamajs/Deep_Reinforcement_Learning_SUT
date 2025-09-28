"""
Visualization and Analysis Utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any


def plot_world_model_training(trainer, title="World Model Training"):
    """Plot world model training losses"""
    plt.figure(figsize=(15, 10))

    # Training losses
    plt.subplot(2, 3, 1)
    plt.plot(trainer.losses["vae_total"], label="VAE Total", linewidth=2)
    plt.plot(trainer.losses["vae_recon"], label="VAE Reconstruction", linewidth=2)
    plt.plot(trainer.losses["vae_kl"], label="VAE KL", linewidth=2)
    plt.title("VAE Training Losses")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(
        trainer.losses["dynamics"], label="Dynamics Loss", color="red", linewidth=2
    )
    plt.title("Dynamics Model Training")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.plot(trainer.losses["reward"], label="Reward Loss", color="green", linewidth=2)
    plt.title("Reward Model Training")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.show()


def plot_rssm_training(trainer, title="RSSM Training"):
    """Plot RSSM training losses"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(trainer.losses["total"], label="Total Loss", linewidth=2)
    plt.title("Total Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(
        trainer.losses["reconstruction"],
        label="Reconstruction",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        trainer.losses["kl_divergence"], label="KL Divergence", color="red", linewidth=2
    )
    plt.plot(trainer.losses["reward"], label="Reward", color="green", linewidth=2)
    plt.title("Component Losses")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    final_losses = [
        trainer.losses["reconstruction"][-1],
        trainer.losses["kl_divergence"][-1],
        trainer.losses["reward"][-1],
    ]
    labels = ["Reconstruction", "KL Divergence", "Reward"]
    bars = plt.bar(labels, final_losses, alpha=0.7, color=["blue", "red", "green"])
    plt.title("Final Performance")
    plt.ylabel("Loss Value")
    plt.yscale("log")

    for bar, loss in zip(bars, final_losses):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{loss:.3f}",
            ha="center",
            va="bottom",
        )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.show()


def plot_dreamer_training(dreamer_agent, training_rewards, title="Dreamer Training"):
    """Plot Dreamer agent training results"""
    plt.figure(figsize=(20, 15))

    # Training progress
    plt.subplot(3, 4, 1)
    plt.plot(training_rewards, alpha=0.7, linewidth=1)
    if len(training_rewards) > 10:
        smooth_rewards = pd.Series(training_rewards).rolling(window=10).mean()
        plt.plot(smooth_rewards, linewidth=2, label="Smooth")
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Actor and critic losses
    plt.subplot(3, 4, 2)
    if dreamer_agent.stats["actor_loss"]:
        plt.plot(dreamer_agent.stats["actor_loss"], label="Actor Loss", linewidth=2)
        plt.plot(dreamer_agent.stats["critic_loss"], label="Critic Loss", linewidth=2)
        plt.title("Actor-Critic Losses")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Imagination rewards
    plt.subplot(3, 4, 3)
    if dreamer_agent.stats["imagination_reward"]:
        plt.plot(dreamer_agent.stats["imagination_reward"], color="purple", linewidth=2)
        plt.title("Imagination Rewards")
        plt.xlabel("Training Step")
        plt.ylabel("Mean Imagined Reward")
        plt.grid(True, alpha=0.3)

    # Policy entropy evolution
    plt.subplot(3, 4, 4)
    if dreamer_agent.stats["policy_entropy"]:
        plt.plot(dreamer_agent.stats["policy_entropy"], color="green", linewidth=2)
        plt.title("Policy Entropy")
        plt.xlabel("Training Step")
        plt.ylabel("Entropy")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.show()


def plot_world_model_analysis(
    world_model, test_data, device, title="World Model Analysis"
):
    """Analyze world model performance"""
    world_model.eval()

    # Test on sample data
    test_batch_size = min(100, len(test_data["observations"]))
    test_indices = torch.randperm(len(test_data["observations"]))[:test_batch_size]

    test_obs = test_data["observations"][test_indices].to(device)
    test_actions = test_data["actions"][test_indices].to(device)
    test_rewards = test_data["rewards"][test_indices].to(device)
    test_next_obs = test_data["next_observations"][test_indices].to(device)

    with torch.no_grad():
        # Test VAE reconstruction
        recon_obs, _, _, z_obs = world_model.vae(test_obs)
        recon_error = F.mse_loss(recon_obs, test_obs).item()

        # Test dynamics prediction
        if world_model.dynamics.stochastic:
            z_pred, _, _ = world_model.dynamics(z_obs, test_actions)
        else:
            z_pred = world_model.dynamics(z_obs, test_actions)

        # Compare predicted latent states with actual
        recon_obs, _, _, z_next_actual = world_model.vae(test_next_obs)
        dynamics_error = F.mse_loss(z_pred, z_next_actual).item()

        # Test reward prediction
        pred_rewards = world_model.reward_model(z_obs, test_actions)
        reward_error = F.mse_loss(pred_rewards, test_rewards).item()

    plt.figure(figsize=(15, 10))

    # Reconstruction visualization
    plt.subplot(2, 3, 1)
    sample_idx = 0
    original_obs = test_obs[sample_idx].cpu().numpy()
    reconstructed_obs = recon_obs[sample_idx].cpu().numpy()

    x_pos = np.arange(len(original_obs))
    width = 0.35

    plt.bar(x_pos - width / 2, original_obs, width, label="Original", alpha=0.7)
    plt.bar(
        x_pos + width / 2, reconstructed_obs, width, label="Reconstructed", alpha=0.7
    )
    plt.title("VAE Reconstruction Example")
    plt.xlabel("State Dimension")
    plt.ylabel("Value")
    plt.legend()

    # Latent space visualization
    plt.subplot(2, 3, 2)
    latent_states = z_obs.cpu().numpy()
    plt.scatter(latent_states[:, 0], latent_states[:, 1], alpha=0.6, s=30)
    plt.title("Latent Space Representation")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True, alpha=0.3)

    # Prediction accuracy
    plt.subplot(2, 3, 3)
    errors = [recon_error, dynamics_error, reward_error]
    labels = ["Reconstruction", "Dynamics", "Reward"]
    colors = ["blue", "red", "green"]

    bars = plt.bar(labels, errors, color=colors, alpha=0.7)
    plt.title("Prediction Errors")
    plt.ylabel("Mean Squared Error")
    plt.yscale("log")

    # Add value labels on bars
    for bar, error in zip(bars, errors):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{error:.2e}",
            ha="center",
            va="bottom",
        )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.show()


def plot_performance_comparison(
    eval_rewards, random_rewards, title="Performance Comparison"
):
    """Plot performance comparison between methods"""
    plt.figure(figsize=(12, 8))

    # Performance comparison
    plt.subplot(2, 2, 1)
    methods = ["Dreamer Agent", "Random Policy"]
    mean_rewards = [np.mean(eval_rewards), np.mean(random_rewards)]
    std_rewards = [np.std(eval_rewards), np.std(random_rewards)]

    bars = plt.bar(
        methods,
        mean_rewards,
        yerr=std_rewards,
        capsize=5,
        alpha=0.7,
        color=["skyblue", "orange"],
    )
    plt.title("Performance Comparison")
    plt.ylabel("Episode Reward")
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, mean_val, std_val in zip(bars, mean_rewards, std_rewards):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_val,
            f"{mean_val:.1f}±{std_val:.1f}",
            ha="center",
            va="bottom",
        )

    # Reward distribution
    plt.subplot(2, 2, 2)
    plt.boxplot(
        [eval_rewards, random_rewards], labels=["Dreamer Agent", "Random Policy"]
    )
    plt.title("Reward Distribution")
    plt.ylabel("Episode Reward")
    plt.grid(True, alpha=0.3)

    # Improvement metrics
    plt.subplot(2, 2, 3)
    improvement = (
        (np.mean(eval_rewards) - np.mean(random_rewards))
        / abs(np.mean(random_rewards))
        * 100
    )
    stability = 1.0 - (np.std(eval_rewards) / abs(np.mean(eval_rewards)))

    metrics = ["Performance\nImprovement (%)", "Training\nStability"]
    values = [improvement, stability]

    bars = plt.bar(metrics, values, alpha=0.7, color=["green", "blue"])
    plt.title("Key Metrics")
    plt.ylabel("Value")

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
        )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.show()
