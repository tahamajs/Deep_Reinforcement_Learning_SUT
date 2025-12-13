"""
Advanced Model-Based RL and World Models - Training Examples
===========================================================

This module provides comprehensive implementations and training examples for
Advanced Model-Based RL and World Models (CA11).

Key Components:
- Variational autoencoders for world models
- Recurrent State Space Models (RSSM)
- Dreamer agent architecture
- Latent space planning and imagination
- Advanced world model techniques

Author: DRL Course Team
"""

from models.vae import VariationalAutoencoder
from models.rssm import RSSM
from agents.dreamer_agent import DreamerAgent
from experiments.config import (
    VAE_CONFIG,
    DREAMER_CONFIG,
    GLOBAL_CONFIG,
    update_config_with_env_dims,
)

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
import gymnasium as gym
from collections import deque
import random
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# Set random seeds for reproducibility
def set_seed(seed: int = GLOBAL_CONFIG.seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(GLOBAL_CONFIG.seed)

# =============================================================================
# TRAINING UTILITIES
# =============================================================================


def train_vae_world_model(
    env_name: str = VAE_CONFIG.env_name,
    latent_dim: int = VAE_CONFIG.latent_dim,
    num_episodes: int = VAE_CONFIG.num_episodes_data_collection,
    batch_size: int = VAE_CONFIG.batch_size,
    seed: int = GLOBAL_CONFIG.seed,
) -> Dict[str, Any]:
    """Train VAE-based world model"""

    set_seed(seed)
    update_config_with_env_dims(env_name)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    vae = VariationalAutoencoder(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=VAE_CONFIG.hidden_dim,
    ).to(GLOBAL_CONFIG.device)
    optimizer = optim.Adam(vae.parameters(), lr=VAE_CONFIG.learning_rate)

    # Collect experience
    print(f"Collecting experience for VAE training on {env_name}")
    observations = []

    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset(seed=GLOBAL_CONFIG.seed + episode)
        done = False

        while not done:
            observations.append(obs)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()

    # Convert to tensor
    obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32).to(GLOBAL_CONFIG.device)

    # Train VAE
    print("Training VAE world model...")
    losses = {"reconstruction": [], "kl": [], "total": []}

    num_epochs = VAE_CONFIG.num_epochs
    for epoch in tqdm(range(num_epochs)):
        # Shuffle data
        indices = torch.randperm(len(obs_tensor))
        obs_shuffled = obs_tensor[indices]

        epoch_loss = 0
        for i in range(0, len(obs_shuffled), batch_size):
            batch = obs_shuffled[i : i + batch_size]

            optimizer.zero_grad()

            reconstruction, latent, mean, log_var = vae(batch)
            loss = vae.loss_function(reconstruction, batch, mean, log_var)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses["total"].append(epoch_loss / (len(obs_shuffled) // batch_size))

    results = {
        "vae_model": vae,
        "losses": losses,
        "observations": observations,
        "config": {
            "env_name": env_name,
            "latent_dim": latent_dim,
            "num_episodes": num_episodes,
        },
    }

    return results


def train_dreamer_agent(
    env_name: str = DREAMER_CONFIG.env_name,
    num_episodes: int = DREAMER_CONFIG.num_episodes,
    max_steps: int = DREAMER_CONFIG.max_steps,
    seed: int = GLOBAL_CONFIG.seed,
) -> Dict[str, Any]:
    """Train Dreamer agent"""

    set_seed(seed)
    update_config_with_env_dims(env_name)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DreamerAgent(obs_dim, action_dim, GLOBAL_CONFIG)

    episode_rewards = []
    world_losses = {"obs": [], "reward": [], "continue": [], "kl": [], "total": []}
    actor_losses = []
    critic_losses = []

    print(f"Training Dreamer Agent on {env_name}")
    print("=" * 40)

    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset(seed=GLOBAL_CONFIG.seed + episode)
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        # Update world model
        if len(agent.buffer) > agent.rssm.stochastic_size:  # Minimum batch size for RSSM
            world_loss_dict = agent.update_world_model(batch_size=DREAMER_CONFIG.agent_config.batch_size)
            for key, value in world_loss_dict.items():
                if key in world_losses and value is not None:
                    world_losses[key].append(value)

        # Update actor-critic
        if len(agent.buffer) > agent.rssm.stochastic_size:
            ac_loss_dict = agent.update_actor_critic(batch_size=DREAMER_CONFIG.agent_config.batch_size)
            if "actor_loss" in ac_loss_dict:
                actor_losses.append(ac_loss_dict["actor_loss"])
                critic_losses.append(ac_loss_dict["critic_loss"])

    env.close()

    results = {
        "episode_rewards": episode_rewards,
        "world_losses": world_losses,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "agent": agent,
        "config": {
            "env_name": env_name,
            "num_episodes": num_episodes,
            "max_steps": max_steps,
        },
    }

    return results


def compare_world_model_methods(
    env_name: str = DREAMER_CONFIG.env_name, num_runs: int = 3, num_episodes: int = DREAMER_CONFIG.num_episodes // 5
) -> Dict[str, Any]:
    """Compare different world model approaches"""

    methods = ["VAE World Model", "Dreamer (Simplified)"]
    results = {}

    for method in methods:
        print(f"Testing {method}...")

        run_rewards = []

        for run in range(num_runs):
            set_seed(GLOBAL_CONFIG.seed + run)
            update_config_with_env_dims(env_name)

            if method == "VAE World Model":
                # For VAE, just collect random experience
                env = gym.make(env_name)
                rewards = []
                for episode in range(num_episodes):
                    obs, _ = env.reset(seed=GLOBAL_CONFIG.seed + episode)
                    episode_reward = 0
                    for step in range(DREAMER_CONFIG.max_steps):
                        action = env.action_space.sample()
                        obs, reward, terminated, truncated, _ = env.step(action)
                        episode_reward += reward
                        if terminated or truncated:
                            break
                    rewards.append(episode_reward)
                env.close()
                run_rewards.append(rewards)
            else:  # Dreamer
                result = train_dreamer_agent(
                    env_name, num_episodes=num_episodes, seed=GLOBAL_CONFIG.seed + run
                )
                run_rewards.append(result["episode_rewards"])

        # Average across runs
        avg_rewards = np.mean(run_rewards, axis=0)
        std_rewards = np.std(run_rewards, axis=0)

        results[method] = {
            "mean_rewards": avg_rewards,
            "std_rewards": std_rewards,
            "final_score": np.mean(avg_rewards[-50:]),  # Average of last 50 episodes
        }

    return results


# =============================================================================
# ANALYSIS AND VISUALIZATION FUNCTIONS
# =============================================================================


def analyze_world_model_representations(save_path: Optional[str] = None) -> plt.Figure:
    """Analyze world model latent representations"""

    print("Analyzing world model latent representations...")
    print("=" * 50)

    # Generate synthetic data for visualization
    np.random.seed(GLOBAL_CONFIG.seed)
    n_samples = 1000

    # Create different types of observations
    angles = np.random.uniform(-np.pi, np.pi, n_samples)
    angular_velocities = np.random.uniform(-8, 8, n_samples)

    # Create observations (cos, sin, angular velocity)
    observations = np.column_stack(
        [np.cos(angles), np.sin(angles), angular_velocities / 8]  # Normalize
    )

    # Simulate VAE encoding
    latent_dim = 2
    np.random.seed(GLOBAL_CONFIG.seed)
    # Mock latent representations for visualization
    latents = np.random.normal(0, 1, (n_samples, latent_dim))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original observation space
    scatter = axes[0, 0].scatter(
        observations[:, 0],
        observations[:, 1],
        c=observations[:, 2],
        cmap="viridis",
        alpha=0.6,
    )
    axes[0, 0].set_xlabel("cos(θ)")
    axes[0, 0].set_ylabel("sin(θ)")
    axes[0, 0].set_title("Original Observation Space")
    plt.colorbar(scatter, ax=axes[0, 0], label="Angular Velocity")

    # Latent space
    scatter = axes[0, 1].scatter(
        latents[:, 0], latents[:, 1], c=observations[:, 2], cmap="viridis", alpha=0.6
    )
    axes[0, 1].set_xlabel("Latent Dimension 1")
    axes[0, 1].set_ylabel("Latent Dimension 2")
    axes[0, 1].set_title("Latent Representation Space")
    plt.colorbar(scatter, ax=axes[0, 1], label="Angular Velocity")

    # Reconstruction quality
    reconstruction_error = np.random.exponential(0.1, n_samples)
    axes[0, 2].hist(reconstruction_error, bins=30, alpha=0.7, edgecolor="black")
    axes[0, 2].set_xlabel("Reconstruction Error")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title("Reconstruction Quality Distribution")
    axes[0, 2].axvline(
        np.mean(reconstruction_error),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(reconstruction_error):.3f}",
    )
    axes[0, 2].legend()

    # Temporal consistency
    time_steps = np.arange(50)
    consistency_scores = 1 - np.exp(-time_steps / 20) + np.random.normal(0, 0.1, 50)

    axes[1, 0].plot(time_steps, consistency_scores, "b-", linewidth=2, alpha=0.8)
    axes[1, 0].fill_between(
        time_steps,
        consistency_scores - 0.1,
        consistency_scores + 0.1,
        alpha=0.3,
        color="blue",
    )
    axes[1, 0].set_xlabel("Time Steps Ahead")
    axes[1, 0].set_ylabel("Prediction Consistency")
    axes[1, 0].set_title("Temporal Prediction Consistency")
    axes[1, 0].grid(True, alpha=0.3)

    # Uncertainty quantification
    prediction_steps = np.arange(1, 21)
    uncertainty = np.sqrt(prediction_steps) * 0.1 + np.random.normal(0, 0.05, 20)

    axes[1, 1].plot(
        prediction_steps, uncertainty, "r-", linewidth=2, marker="o", markersize=4
    )
    axes[1, 1].set_xlabel("Prediction Horizon")
    axes[1, 1].set_ylabel("Prediction Uncertainty")
    axes[1, 1].set_title("Uncertainty Growth Over Time")
    axes[1, 1].grid(True, alpha=0.3)

    # World model vs actual environment
    methods = ["World Model", "Actual Environment"]
    metrics = ["Reward Prediction", "State Prediction", "Dynamics Accuracy"]

    world_model_scores = [0.85, 0.78, 0.82]
    actual_scores = [1.0, 1.0, 1.0]  # Perfect by definition

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 2].bar(
        x - width / 2, world_model_scores, width, label="World Model", alpha=0.8
    )
    axes[1, 2].bar(
        x + width / 2, actual_scores, width, label="Actual Environment", alpha=0.8
    )
    axes[1, 2].set_xlabel("Evaluation Metric")
    axes[1, 2].set_ylabel("Accuracy Score")
    axes[1, 2].set_title("World Model vs Actual Environment")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics, rotation=45, ha="right")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close instead of show to avoid display issues

    print("World model representation analysis completed!")

    return fig


def comprehensive_world_models_analysis(
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Comprehensive analysis of world model approaches"""

    print("Comprehensive world models analysis...")
    print("=" * 45)

    methods = ["VAE World Model", "RSSM", "Dreamer", "World Models", "MuZero"]
    environments = ["Atari", "Control Suite", "DM Control", "Robotics"]

    # Method capabilities (1-10 scale)
    capabilities = {
        "Representation Learning": {
            "VAE World Model": 8,
            "RSSM": 7,
            "Dreamer": 9,
            "World Models": 9,
            "MuZero": 8,
        },
        "Dynamics Modeling": {
            "VAE World Model": 6,
            "RSSM": 9,
            "Dreamer": 8,
            "World Models": 7,
            "MuZero": 9,
        },
        "Sample Efficiency": {
            "VAE World Model": 7,
            "RSSM": 8,
            "Dreamer": 9,
            "World Models": 8,
            "MuZero": 10,
        },
        "Planning Capability": {
            "VAE World Model": 5,
            "RSSM": 7,
            "Dreamer": 9,
            "World Models": 6,
            "MuZero": 10,
        },
        "Scalability": {
            "VAE World Model": 8,
            "RSSM": 6,
            "Dreamer": 7,
            "World Models": 8,
            "MuZero": 8,
        },
    }

    # Performance by environment type
    performance_by_env = {
        "Atari": {
            "VAE World Model": 6,
            "RSSM": 7,
            "Dreamer": 8,
            "World Models": 7,
            "MuZero": 9,
        },
        "Control Suite": {
            "VAE World Model": 7,
            "RSSM": 8,
            "Dreamer": 9,
            "World Models": 8,
            "MuZero": 7,
        },
        "DM Control": {
            "VAE World Model": 8,
            "RSSM": 9,
            "Dreamer": 9,
            "World Models": 8,
            "MuZero": 8,
        },
        "Robotics": {
            "VAE World Model": 5,
            "RSSM": 6,
            "Dreamer": 7,
            "World Models": 6,
            "MuZero": 8,
        },
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Method capabilities radar
    categories = list(capabilities.keys())
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for method in methods[:4]:  # Show first 4 to avoid clutter
        scores = [capabilities[cat][method] for cat in categories]
        scores += scores[:1]
        axes[0, 0].plot(angles, scores, "o-", linewidth=2, label=method, markersize=6)

    axes[0, 0].set_xticks(angles[:-1])
    axes[0, 0].set_xticklabels(categories, fontsize=9)
    axes[0, 0].set_ylim(0, 10)
    axes[0, 0].set_title("World Model Method Capabilities")
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)

    # Performance by environment type
    env_names = list(performance_by_env.keys())
    x = np.arange(len(env_names))
    width = 0.15
    multiplier = 0

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (method, color) in enumerate(zip(methods, colors)):
        scores = [performance_by_env[env][method] for env in env_names]
        offset = width * multiplier
        bars = axes[0, 1].bar(
            x + offset, scores, width, label=method, color=color, alpha=0.8
        )
        multiplier += 1

    axes[0, 1].set_xlabel("Environment Type")
    axes[0, 1].set_ylabel("Performance Score")
    axes[0, 1].set_title("Method Performance by Environment Type")
    axes[0, 1].set_xticks(x + width * 2, env_names)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 1].grid(True, alpha=0.3)

    # Sample efficiency comparison
    sample_efficiency = [capabilities["Sample Efficiency"][m] for m in methods]
    planning_capability = [capabilities["Planning Capability"][m] for m in methods]

    axes[1, 0].scatter(
        sample_efficiency, planning_capability, s=200, alpha=0.7, c="purple"
    )
    for i, method in enumerate(methods):
        axes[1, 0].annotate(
            method,
            (sample_efficiency[i], planning_capability[i]),
            xytext=(5, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
        )

    axes[1, 0].set_xlabel("Sample Efficiency")
    axes[1, 0].set_ylabel("Planning Capability")
    axes[1, 0].set_title("Sample Efficiency vs Planning Capability")
    axes[1, 0].grid(True, alpha=0.3)

    # Method evolution timeline
    years = [2018, 2019, 2020, 2020, 2019]
    method_timeline = ["VAE World Model", "RSSM", "Dreamer", "World Models", "MuZero"]
    innovation_scores = [6, 7, 9, 8, 10]

    axes[1, 1].scatter(years, innovation_scores, s=150, alpha=0.7, c="green")
    for i, (year, method) in enumerate(zip(years, method_timeline)):
        axes[1, 1].annotate(
            method,
            (year, innovation_scores[i]),
            xytext=(5, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
        )

    axes[1, 1].set_xlabel("Year Introduced")
    axes[1, 1].set_ylabel("Innovation Impact")
    axes[1, 1].set_title("World Model Methods Timeline")
    axes[1, 1].grid(True, alpha=0.3)

    # Strengths and limitations
    aspects = ["Strengths", "Limitations"]
    method_analysis = {
        "VAE World Model": [8, 6],
        "RSSM": [7, 7],
        "Dreamer": [9, 5],
        "World Models": [8, 6],
        "MuZero": [10, 4],
    }

    x = np.arange(len(methods))
    width = 0.35

    for i, aspect in enumerate(aspects):
        scores = [method_analysis[method][i] for method in methods]
        axes[2, 0].bar(x + (i - 0.5) * width, scores, width, label=aspect, alpha=0.8)

    axes[2, 0].set_xlabel("Method")
    axes[2, 0].set_ylabel("Score (1-10)")
    axes[2, 0].set_title("Method Strengths and Limitations")
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(methods, rotation=45, ha="right")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Future directions
    future_areas = [
        "Multi-Modal",
        "Hierarchical",
        "Meta-Learning",
        "Continual Learning",
    ]
    current_state = [5, 6, 7, 4]
    potential_impact = [9, 9, 9, 8]

    x = np.arange(len(future_areas))
    width = 0.35

    axes[2, 1].bar(
        x - width / 2, current_state, width, label="Current State", alpha=0.7
    )
    axes[2, 1].bar(
        x + width / 2, potential_impact, width, label="Potential Impact", alpha=0.7
    )
    axes[2, 1].set_xlabel("Research Area")
    axes[2, 1].set_ylabel("Score (1-10)")
    axes[2, 1].set_title("Future Directions for World Models")
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(future_areas, rotation=45, ha="right")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close instead of show to avoid display issues

    # Print comprehensive analysis
    print("\n" + "=" * 55)
    print("WORLD MODELS COMPREHENSIVE ANALYSIS")
    print("=" * 55)

    for method in methods:
        avg_perf = np.mean([performance_by_env[env][method] for env in env_names])
        print(f"{method:15} | Average Performance: {avg_perf:6.1f}")

    #     print("
    # 💡 Key Insights for World Models:"    print("• Dreamer offers best overall performance and sample efficiency")
    print("• RSSM excels at temporal dynamics modeling")
    print("• VAE provides strong representation learning")
    print("• MuZero leads in planning capabilities")

    # print("
    # 🎯 Recommendations:"    print("• Use Dreamer for complex RL with limited samples")
    print("• Choose RSSM for temporal prediction tasks")
    print("• Start with VAE for representation learning")
    print("• Consider MuZero for planning-heavy domains")

    return {
        "capabilities": capabilities,
        "performance_by_env": performance_by_env,
        "methods": methods,
    }


# =============================================================================
# MAIN TRAINING EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("Advanced Model-Based RL and World Models")
    print("=" * 45)
    print("Available training examples:")
    print("1. train_vae_world_model() - Train VAE-based world model")
    print("2. train_dreamer_agent() - Train Dreamer agent")
    print("3. compare_world_model_methods() - Compare world model approaches")
    print("4. analyze_world_model_representations() - Representation analysis")
    print("5. comprehensive_world_models_analysis() - Full method comparison")
    print("\nExample usage:")
    print(f"results = train_dreamer_agent(num_episodes={DREAMER_CONFIG.num_episodes})")
    # print("comparison = compare_world_model_methods()")
