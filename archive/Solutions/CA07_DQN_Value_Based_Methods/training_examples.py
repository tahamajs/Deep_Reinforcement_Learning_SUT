"""
Deep Q-Networks (DQN) and Value-Based Methods - Training Examples and Implementations
Computer Assignment 7 - Sharif University of Technology
Deep Reinforcement Learning Course

This module provides comprehensive implementations of DQN variants including
Vanilla DQN, Double DQN, Dueling DQN, and advanced analysis tools.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import os

from src.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, DuelingDoubleDQNAgent, NoisyDQNAgent
from src.config import DQNConfig
from src.utils import set_seed, smooth_curve # Assuming these will be moved to src/utils

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
set_seed(DQNConfig.SEED)


# Training Functions


def train_dqn_agent(
    agent_class: type,
    env_name: str = DQNConfig.ENV_NAME,
    episodes: int = DQNConfig.TRAIN_EPISODES,
    config: DQNConfig = DQNConfig(),
) -> Dict[str, List[float]]:
    """
    Trains a DQN agent for a specified number of episodes.

    Args:
        agent_class: The class of the DQN agent to train (e.g., DQNAgent, DoubleDQNAgent).
        env_name: The name of the Gymnasium environment to use.
        episodes: The total number of episodes to train for.
        config: The DQNConfig object containing hyperparameters.

    Returns:
        A dictionary containing training metrics: scores, losses, and epsilon history.
    """

    print(f"Training {agent_class.__name__} on {env_name}")
    print("=" * 50)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = agent_class(state_dim=state_dim, action_dim=action_dim, config=config)

    scores = []
    avg_losses = []

    for episode in range(episodes):
        reward, info = agent.train_episode(env, max_steps=config.MAX_EPISODE_STEPS)
        scores.append(reward)
        avg_losses.append(info["avg_loss"])

        if (episode + 1) % config.LOG_INTERVAL == 0:
            avg_score = np.mean(scores[-config.LOG_INTERVAL:])
            avg_loss = np.mean(avg_losses[-config.LOG_INTERVAL:])
            print(
                f"Episode {episode+1:3d} | Average Score: {avg_score:6.1f} | Average Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}"
            )

    env.close()
    return {
        "scores": scores,
        "losses": avg_losses,
        "epsilon_history": agent.epsilon_history,
    }


def compare_dqn_variants(
    env_name: str = DQNConfig.ENV_NAME,
    episodes: int = DQNConfig.TRAIN_EPISODES,
    config: DQNConfig = DQNConfig(),
) -> Dict[str, Dict]:
    """
    Compares the performance of different DQN variants.

    Args:
        env_name: The name of the Gymnasium environment to use.
        episodes: The number of episodes to train each variant for.
        config: The DQNConfig object containing hyperparameters.

    Returns:
        A dictionary where keys are agent names and values are their training results.
    """

    print(f"Comparing DQN Variants on {env_name}")
    print("=" * 45)

    variants = {
        "Vanilla DQN": DQNAgent,
        "Double DQN": DoubleDQNAgent,
        "Dueling DQN": DuelingDQNAgent,
        "Dueling Double DQN": DuelingDoubleDQNAgent,
        "Noisy DQN": NoisyDQNAgent,
    }

    results = {}

    for name, agent_class in variants.items():
        print(f"\nTraining {name}...")
        result = train_dqn_agent(agent_class, env_name, episodes, config=config)
        results[name] = result

    return results


def plot_dqn_comparison(results: Dict[str, Dict], save_path: Optional[str] = None):
    """
    Plots a comparison of learning curves, loss curves, and final performance
    for different DQN variants.

    Args:
        results: A dictionary containing training results for each DQN variant.
        save_path: Optional path to save the generated plot.
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    methods = list(results.keys())
    colors = ["blue", "green", "red", "purple", "orange"] # Added color for Noisy DQN

    # Learning curves
    for method, color in zip(methods, colors):
        scores = results[method]["scores"]
        smoothed_scores = smooth_curve(scores)
        axes[0, 0].plot(smoothed_scores, label=method, color=color, linewidth=2)

    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Smoothed Score")
    axes[0, 0].set_title("Learning Curves Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Loss curves
    for method, color in zip(methods, colors):
        losses = results[method]["losses"]
        smoothed_losses = smooth_curve(losses)
        axes[0, 1].plot(smoothed_losses, label=method, color=color, linewidth=2)

    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Smoothed Loss")
    axes[0, 1].set_title("Loss Curves Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Final performance comparison
    final_scores = [np.mean(results[method]["scores"][-50:]) for method in methods]
    axes[1, 0].bar(methods, final_scores, alpha=0.7, edgecolor="black", color=colors)
    axes[1, 0].set_ylabel("Final Average Score")
    axes[1, 0].set_title("Final Performance Comparison")
    axes[1, 0].grid(True, alpha=0.3)

    # Training stability (variance of scores)
    score_variances = [np.var(results[method]["scores"][-100:]) for method in methods]
    axes[1, 1].bar(
        methods, score_variances, alpha=0.7, edgecolor="black", color=colors
    )
    axes[1, 1].set_ylabel("Score Variance")
    axes[1, 1].set_title("Training Stability (Lower is Better)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def hyperparameter_optimization_study(
    env_name: str = DQNConfig.ENV_NAME,
    episodes: int = DQNConfig.TRAIN_EPISODES,
    config: DQNConfig = DQNConfig(),
):
    """
    Conducts a hyperparameter optimization study for DQN agents.

    Examines the impact of different network architectures and exploration schedules
    on agent performance.

    Args:
        env_name: The name of the Gymnasium environment to use.
        episodes: The number of episodes to train each configuration for.
        config: The DQNConfig object containing baseline hyperparameters.

    Returns:
        A dictionary containing results for architecture and exploration studies.
    """

    print("DQN Hyperparameter Optimization Study")
    print("=" * 45)

    # Test different architectures
    architectures = [
        {"hidden_dim": 64, "lr": 1e-3},
        {"hidden_dim": 128, "lr": 1e-3},
        {"hidden_dim": 256, "lr": 1e-3},
        {"hidden_dim": 128, "lr": 5e-4},
        {"hidden_dim": 128, "lr": 2e-3},
    ]

    arch_results = {}

    print("\nTesting different architectures...")
    for i, arch in enumerate(architectures):
        print(f"  Architecture {i+1}: Hidden={arch['hidden_dim']}, LR={arch['lr']}")
        current_config = DQNConfig()
        current_config.HIDDEN_DIM = arch["hidden_dim"]
        current_config.LR = arch["lr"]

        result = train_dqn_agent(
            DoubleDQNAgent,
            env_name,
            episodes,
            config=current_config,
        )
        final_score = np.mean(result["scores"][-30:])
        arch_results[f"H{arch['hidden_dim']}_LR{arch['lr']}"] = final_score

    # Test different exploration schedules
    exploration_schedules = [
        {"eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.995},
        {"eps_start": 1.0, "eps_end": 0.1, "eps_decay": 0.99},
        {"eps_start": 0.5, "eps_end": 0.01, "eps_decay": 0.995},
        {"eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.999},
    ]

    exploration_results = {}

    print("\nTesting different exploration schedules...")
    for i, sched in enumerate(exploration_schedules):
        print(
            f"  Schedule {i+1}: Start={sched['eps_start']}, End={sched['eps_end']}, Decay={sched['eps_decay']}"
        )
        current_config = DQNConfig()
        current_config.EPSILON_START = sched["eps_start"]
        current_config.EPSILON_END = sched["eps_end"]
        current_config.EPSILON_DECAY = sched["eps_decay"]

        result = train_dqn_agent(
            DoubleDQNAgent,
            env_name,
            episodes,
            config=current_config,
        )
        final_score = np.mean(result["scores"][-30:])
        exploration_results[f"S{i+1}"] = final_score

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Architecture comparison
    arch_names = list(arch_results.keys())
    arch_scores = list(arch_results.values())
    axes[0].bar(arch_names, arch_scores, alpha=0.7, edgecolor="black")
    axes[0].set_ylabel("Final Average Score")
    axes[0].set_title("Architecture Comparison")
    axes[0].set_xticklabels(arch_names, rotation=45, ha="right")
    axes[0].grid(True, alpha=0.3)

    # Exploration schedule comparison
    exp_names = list(exploration_results.keys())
    exp_scores = list(exploration_results.values())
    axes[1].bar(exp_names, exp_scores, alpha=0.7, edgecolor="black", color="green")
    axes[1].set_ylabel("Final Average Score")
    axes[1].set_title("Exploration Schedule Comparison")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.VISUALIZATIONS_DIR, "dqn_hyperparameter_optimization.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {"architectures": arch_results, "exploration_schedules": exploration_results}


def robustness_analysis(
    env_name: str = DQNConfig.ENV_NAME,
    episodes: int = DQNConfig.TRAIN_EPISODES,
    config: DQNConfig = DQNConfig(),
):
    """
    Analyzes the robustness of DQN variants to different conditions, such as
    varying random seeds and reward scaling.

    Args:
        env_name: The name of the Gymnasium environment to use.
        episodes: The number of episodes to train for each condition.
        config: The DQNConfig object containing baseline hyperparameters.

    Returns:
        A dictionary containing robustness analysis results for seeds and reward scales.
    """

    print("DQN Robustness Analysis")
    print("=" * 30)

    # Test on different random seeds
    seeds = [42, 123, 456, 789, 999]
    robustness_results = {}

    print("\nTesting robustness to random seeds...")
    for seed in seeds:
        set_seed(seed) # Use the global set_seed from src/utils

        result = train_dqn_agent(DoubleDQNAgent, env_name, episodes, config=config)
        final_score = np.mean(result["scores"][-30:])
        robustness_results[f"Seed_{seed}"] = final_score

    # Test with different reward scales
    reward_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    scale_results = {}

    print("\nTesting robustness to reward scaling...")
    for scale in reward_scales:
        print(f"  Reward Scale: {scale}")

        # Create custom environment wrapper for reward scaling
        class ScaledRewardEnv(gym.Wrapper):
            def __init__(self, env, scale):
                super().__init__(env)
                self.scale = scale

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                return obs, reward * self.scale, terminated, truncated, info

        env = ScaledRewardEnv(gym.make(env_name), scale)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = DoubleDQNAgent(state_dim, action_dim, config=config)
        scores = []

        for episode in range(episodes):
            reward, _ = agent.train_episode(env, max_steps=config.MAX_EPISODE_STEPS)
            scores.append(reward)

        final_score = np.mean(scores[-30:])
        scale_results[f"Scale_{scale}"] = final_score
        env.close()

    # Plot robustness analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Seed robustness
    seed_names = list(robustness_results.keys())
    seed_scores = list(robustness_results.values())
    axes[0].bar(seed_names, seed_scores, alpha=0.7, edgecolor="black")
    axes[0].set_ylabel("Final Average Score")
    axes[0].set_title("Robustness to Random Seeds")
    axes[0].grid(True, alpha=0.3)

    # Reward scale robustness
    scale_names = list(scale_results.keys())
    scale_scores = list(scale_results.values())
    axes[1].plot(
        reward_scales, scale_scores, "o-", linewidth=2, markersize=8, color="red"
    )
    axes[1].set_xlabel("Reward Scale")
    axes[1].set_ylabel("Final Average Score")
    axes[1].set_title("Robustness to Reward Scaling")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.VISUALIZATIONS_DIR, "dqn_robustness_analysis.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Print robustness statistics
    print("\nRobustness Analysis Results:")
    print("=" * 35)
    print("Random Seed Robustness:")
    print(f"  Mean Score: {np.mean(list(robustness_results.values())):.1f}")
    print(f"  Std Score:  {np.std(list(robustness_results.values())):.1f}")
    print(f"  Min Score:  {np.min(list(robustness_results.values())):.1f}")
    print(f"  Max Score:  {np.max(list(robustness_results.values())):.1f}")

    print("\nReward Scale Robustness:")
    for scale, score in scale_results.items():
        print(f"  {scale}: {score:.1f}")

    return {"seed_robustness": robustness_results, "scale_robustness": scale_results}


def advanced_dqn_training_demo(config: DQNConfig = DQNConfig()):
    """
    Demonstrates advanced DQN training techniques, including concepts from Rainbow DQN.

    Note: The full implementations of Prioritized Experience Replay, N-step Q-learning,
    and Distributional RL are not provided in this demo but are conceptualized.
    This function primarily illustrates their potential impact through simulated results.

    Args:
        config: The DQNConfig object containing hyperparameters.
    """

    print("Advanced DQN Training Techniques Demo")
    print("=" * 45)

    # 1. Prioritized Experience Replay comparison
    print("\n1. Comparing Uniform vs Prioritized Experience Replay (Simulated)...")
    print("   (Requires implementing PrioritizedReplayBuffer)")

    uniform_scores = np.random.normal(180, 20, 100)
    prioritized_scores = np.random.normal(195, 15, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(uniform_scores, alpha=0.7, label="Uniform Replay", bins=20)
    ax.hist(prioritized_scores, alpha=0.7, label="Prioritized Replay", bins=20)
    ax.set_xlabel("Final Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Uniform vs Prioritized Experience Replay (Simulated)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(config.VISUALIZATIONS_DIR, "uniform_vs_prioritized_replay.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # 2. Multi-step learning comparison
    print("\n2. Comparing different n-step returns (Simulated)...")
    print("   (Requires modifying agent's Q-value update for N-step returns)")

    n_steps = [1, 2, 3, 4, 5]
    n_step_scores = [180, 190, 195, 185, 175] # Simulated results

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_steps, n_step_scores, "o-", linewidth=2, markersize=8, color="purple")
    ax.set_xlabel("n-step Returns")
    ax.set_ylabel("Final Average Score")
    ax.set_title("Multi-step Learning Performance (Simulated)")
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(config.VISUALIZATIONS_DIR, "multi_step_learning.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # 3. Distributional RL comparison
    print("\n3. Comparing DQN vs C51 (Distributional RL) (Simulated)...")
    print("   (Requires implementing a CategoricalQNetwork and corresponding loss)")

    dqn_scores = np.random.normal(185, 25, 50)
    c51_scores = np.random.normal(200, 20, 50) # Simulated results

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(dqn_scores, alpha=0.7, label="DQN", bins=15)
    ax.hist(c51_scores, alpha=0.7, label="C51", bins=15)
    ax.set_xlabel("Final Score")
    ax.set_ylabel("Frequency")
    ax.set_title("DQN vs Distributional RL (C51) (Simulated)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(config.VISUALIZATIONS_DIR, "dqn_vs_distributional.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nAdvanced techniques summary (Simulated Results):")
    print("• Prioritized replay improves sample efficiency")
    print("• Multi-step learning (n=3) often optimal")
    print("• Distributional methods provide better value estimation")
    print("• Combining techniques yields best results")


# Main execution examples
if __name__ == "__main__":
    current_config = DQNConfig()
    print("Deep Q-Networks - Training Examples")
    print("=" * 40)

    # Example 1: Compare DQN variants
    print("\nExample 1: Comparing DQN Variants")
    results = compare_dqn_variants(config=current_config)
    plot_dqn_comparison(results, os.path.join(current_config.VISUALIZATIONS_DIR, "dqn_variants_comparison.png"))

    # Example 2: Hyperparameter optimization
    print("\nExample 2: Hyperparameter Optimization Study")
    hyper_results = hyperparameter_optimization_study(config=current_config)

    # Example 3: Robustness analysis
    print("\nExample 3: Robustness Analysis")
    robustness_results = robustness_analysis(config=current_config)

    # Example 4: Advanced techniques demo
    print("\nExample 4: Advanced DQN Techniques Demo")
    advanced_dqn_training_demo(config=current_config)

    print("\nAll examples completed! Check the generated plots and results.")
