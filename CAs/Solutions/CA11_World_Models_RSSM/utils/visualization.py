"""
Visualization Utilities for World Models

This module provides comprehensive visualization functions for analyzing
world models, RSSM, and Dreamer agents.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path


def plot_world_model_training(
    losses: Dict[str, List[float]],
    title: str = "World Model Training Progress",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot world model training losses"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(losses.get("total", []), "b-", linewidth=2)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    
    # VAE loss
    axes[0, 1].plot(losses.get("vae", []), "r-", linewidth=2)
    axes[0, 1].set_title("VAE Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dynamics loss
    axes[1, 0].plot(losses.get("dynamics", []), "g-", linewidth=2)
    axes[1, 0].set_title("Dynamics Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward loss
    axes[1, 1].plot(losses.get("reward", []), "m-", linewidth=2)
    axes[1, 1].set_title("Reward Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_rssm_training(
    losses: Dict[str, List[float]],
    title: str = "RSSM Training Progress",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot RSSM training losses"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(losses.get("total", []), "b-", linewidth=2)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(losses.get("reconstruction", []), "r-", linewidth=2)
    axes[0, 1].set_title("Reconstruction Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reward loss
    axes[1, 0].plot(losses.get("reward", []), "g-", linewidth=2)
    axes[1, 0].set_title("Reward Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)
    
    # KL loss
    axes[1, 1].plot(losses.get("kl", []), "m-", linewidth=2)
    axes[1, 1].set_title("KL Divergence Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_dreamer_training(
    dreamer_agent,
    title: str = "Dreamer Agent Training Progress",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Dreamer agent training statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    stats = dreamer_agent.stats
    
    # Actor loss
    if "actor_loss" in stats and stats["actor_loss"]:
        axes[0, 0].plot(stats["actor_loss"], "b-", linewidth=2)
        axes[0, 0].set_title("Actor Loss")
        axes[0, 0].set_xlabel("Training Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)
    
    # Critic loss
    if "critic_loss" in stats and stats["critic_loss"]:
        axes[0, 1].plot(stats["critic_loss"], "r-", linewidth=2)
        axes[0, 1].set_title("Critic Loss")
        axes[0, 1].set_xlabel("Training Step")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Imagination reward
    if "imagination_reward" in stats and stats["imagination_reward"]:
        axes[1, 0].plot(stats["imagination_reward"], "g-", linewidth=2)
        axes[1, 0].set_title("Imagination Reward")
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("Reward")
        axes[1, 0].grid(True, alpha=0.3)
    
    # Policy entropy
    if "policy_entropy" in stats and stats["policy_entropy"]:
        axes[1, 1].plot(stats["policy_entropy"], "m-", linewidth=2)
        axes[1, 1].set_title("Policy Entropy")
        axes[1, 1].set_xlabel("Training Step")
        axes[1, 1].set_ylabel("Entropy")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_episode_rewards(
    rewards: List[float],
    title: str = "Episode Rewards",
    window_size: int = 50,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot episode rewards with moving average"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Raw rewards
    ax1.plot(rewards, "b-", alpha=0.3, linewidth=1)
    ax1.set_title(f"{title} - Raw")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)
    
    # Moving average
    if len(rewards) >= window_size:
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        ax2.plot(moving_avg, "r-", linewidth=2)
        ax2.set_title(f"{title} - Moving Average (window={window_size})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Reward")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_latent_space(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Latent Space Visualization",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 2D latent space visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    if latents.shape[1] >= 2:
        # 2D scatter plot
        scatter = axes[0].scatter(
            latents[:, 0], 
            latents[:, 1], 
            c=labels if labels is not None else "blue",
            cmap="viridis" if labels is not None else None,
            alpha=0.6
        )
        axes[0].set_xlabel("Latent Dimension 1")
        axes[0].set_ylabel("Latent Dimension 2")
        axes[0].set_title("2D Latent Space")
        axes[0].grid(True, alpha=0.3)
        
        if labels is not None:
            plt.colorbar(scatter, ax=axes[0], label="Label")
    
    # Latent distribution
    if latents.shape[1] >= 2:
        axes[1].hist2d(
            latents[:, 0], 
            latents[:, 1], 
            bins=30, 
            cmap="Blues"
        )
        axes[1].set_xlabel("Latent Dimension 1")
        axes[1].set_ylabel("Latent Dimension 2")
        axes[1].set_title("Latent Distribution")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title: str = "Reconstruction Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot original vs reconstructed observations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot first few dimensions
    num_dims = min(3, original.shape[1])
    
    for i in range(num_dims):
        # Original
        axes[0, i].plot(original[:, i], "b-", linewidth=2, label="Original")
        axes[0, i].plot(reconstructed[:, i], "r--", linewidth=2, label="Reconstructed")
        axes[0, i].set_title(f"Dimension {i+1}")
        axes[0, i].set_xlabel("Time Step")
        axes[0, i].set_ylabel("Value")
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Reconstruction error
        error = np.abs(original[:, i] - reconstructed[:, i])
        axes[1, i].plot(error, "g-", linewidth=2)
        axes[1, i].set_title(f"Reconstruction Error - Dim {i+1}")
        axes[1, i].set_xlabel("Time Step")
        axes[1, i].set_ylabel("Absolute Error")
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_imagination_trajectory(
    imagined_states: np.ndarray,
    imagined_rewards: np.ndarray,
    title: str = "Imagined Trajectory",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot imagined trajectory from world model"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Imagined states
    if imagined_states.shape[1] >= 2:
        axes[0, 0].plot(imagined_states[:, 0], imagined_states[:, 1], "b-o", linewidth=2, markersize=4)
        axes[0, 0].set_xlabel("State Dimension 1")
        axes[0, 0].set_ylabel("State Dimension 2")
        axes[0, 0].set_title("Imagined State Trajectory")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Imagined rewards
    axes[0, 1].plot(imagined_rewards, "r-o", linewidth=2, markersize=4)
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].set_title("Imagined Rewards")
    axes[0, 1].grid(True, alpha=0.3)
    
    # State evolution over time
    if imagined_states.shape[1] >= 2:
        for i in range(min(2, imagined_states.shape[1])):
            axes[1, 0].plot(imagined_states[:, i], label=f"Dim {i+1}", linewidth=2)
        axes[1, 0].set_xlabel("Time Step")
        axes[1, 0].set_ylabel("State Value")
        axes[1, 0].set_title("State Evolution")
        axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 1].hist(imagined_rewards, bins=20, alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("Reward")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Imagined Reward Distribution")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_world_model_comparison(
    results: Dict[str, Any],
    title: str = "World Model Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comparison between different world model methods"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    methods = list(results.keys())
    
    # Episode rewards comparison
    for method in methods:
        if "episode_rewards" in results[method]:
            rewards = results[method]["episode_rewards"]
            moving_avg = pd.Series(rewards).rolling(window=50).mean()
            axes[0, 0].plot(moving_avg, label=method, linewidth=2)
    
    axes[0, 0].set_title("Episode Rewards Comparison")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Final performance comparison
    final_rewards = []
    method_names = []
    for method in methods:
        if "final_avg_reward" in results[method]:
            final_rewards.append(results[method]["final_avg_reward"])
            method_names.append(method)
    
    if final_rewards:
        bars = axes[0, 1].bar(method_names, final_rewards, alpha=0.7)
        axes[0, 1].set_title("Final Performance Comparison")
        axes[0, 1].set_ylabel("Average Reward")
    axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_rewards):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
    
    # Sample efficiency comparison
    sample_efficiency = []
    for method in methods:
        if "episode_rewards" in results[method]:
            rewards = results[method]["episode_rewards"]
            # Calculate episodes to reach 80% of final performance
            final_perf = np.mean(rewards[-50:])
            target_perf = 0.8 * final_perf
            episodes_to_target = len(rewards)
            for i, reward in enumerate(rewards):
                if np.mean(rewards[max(0, i-10):i+1]) >= target_perf:
                    episodes_to_target = i
                    break
            sample_efficiency.append(episodes_to_target)
        else:
            sample_efficiency.append(0)
    
    bars = axes[1, 0].bar(method_names, sample_efficiency, alpha=0.7, color="orange")
    axes[1, 0].set_title("Sample Efficiency (Episodes to 80% Performance)")
    axes[1, 0].set_ylabel("Episodes")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training stability comparison
    stability_scores = []
    for method in methods:
        if "episode_rewards" in results[method]:
            rewards = results[method]["episode_rewards"]
            # Calculate coefficient of variation as stability measure
            if len(rewards) > 100:
                recent_rewards = rewards[-100:]
                stability = 1.0 / (np.std(recent_rewards) / (np.mean(recent_rewards) + 1e-8))
                stability_scores.append(stability)
            else:
                stability_scores.append(0)
        else:
            stability_scores.append(0)
    
    bars = axes[1, 1].bar(method_names, stability_scores, alpha=0.7, color="green")
    axes[1, 1].set_title("Training Stability")
    axes[1, 1].set_ylabel("Stability Score")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def create_comprehensive_report(
    results: Dict[str, Any],
    save_dir: str = "visualizations",
) -> None:
    """Create comprehensive visualization report"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    print("Creating comprehensive visualization report...")
    
    # World model training plots
    for method_name, method_results in results.items():
        if "world_model_losses" in method_results:
            plot_world_model_training(
                method_results["world_model_losses"],
                title=f"{method_name} - World Model Training",
                save_path=save_path / f"{method_name}_world_model_training.png"
            )
        
        if "episode_rewards" in method_results:
            plot_episode_rewards(
                method_results["episode_rewards"],
                title=f"{method_name} - Episode Rewards",
                save_path=save_path / f"{method_name}_episode_rewards.png"
            )
    
    # Comparison plots
    plot_world_model_comparison(
        results,
        title="World Model Methods Comparison",
        save_path=save_path / "methods_comparison.png"
    )
    
    print(f"Visualization report saved to: {save_path}")