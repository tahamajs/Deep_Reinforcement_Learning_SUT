"""
DQN Utilities and Analysis Tools
================================

This module contains utility functions and analysis tools for DQN experiments,
including visualization, performance analysis, and debugging utilities.

Author: CA7 Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Any, Optional
import warnings

# Set plotting style
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
warnings.filterwarnings("ignore")


class QNetworkVisualization:
    """
    Visualization tools for understanding Q-learning concepts
    """

    def __init__(self):
        self.fig_count = 0

    def visualize_q_learning_concepts(self):
        """Visualize core Q-learning concepts"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Q-learning Update Mechanism
        ax = axes[0, 0]

        # Simulate Q-values for a simple grid world
        states = ["S1", "S2", "S3", "S4"]
        actions = ["Up", "Down", "Left", "Right"]

        # Sample Q-values before and after update
        q_before = np.random.rand(4, 4) * 10
        q_after = q_before.copy()
        q_after[1, 2] += 2  # Simulate an update

        # Create heatmap comparison
        im1 = ax.imshow(q_before, cmap="viridis", aspect="auto")
        ax.set_title("Q-Values Before Update")
        ax.set_xticks(range(4))
        ax.set_xticklabels(actions)
        ax.set_yticks(range(4))
        ax.set_yticklabels(states)

        # Add values as text
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{q_before[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                )

        plt.colorbar(im1, ax=ax)

        # 2. Experience Replay Concept
        ax = axes[0, 1]

        # Simulate sequential vs random sampling
        episodes = np.arange(1, 101)
        sequential_loss = 10 * np.exp(-episodes / 30) + np.random.normal(0, 0.5, 100)
        replay_loss = 8 * np.exp(-episodes / 20) + np.random.normal(0, 0.3, 100)

        ax.plot(
            episodes,
            sequential_loss,
            label="Sequential Training",
            alpha=0.7,
            linewidth=2,
        )
        ax.plot(
            episodes, replay_loss, label="Experience Replay", alpha=0.7, linewidth=2
        )
        ax.fill_between(episodes, sequential_loss, alpha=0.3)
        ax.fill_between(episodes, replay_loss, alpha=0.3)

        ax.set_title("Learning Curves: Sequential vs Replay")
        ax.set_xlabel("Training Episodes")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Target Network Stability
        ax = axes[1, 0]

        steps = np.arange(0, 1000)
        # Main network updates frequently
        main_q = 10 + np.cumsum(np.random.normal(0, 0.1, 1000))
        # Target network updates every 100 steps
        target_q = []
        current_target = 10

        for i, step in enumerate(steps):
            if step % 100 == 0 and step > 0:
                current_target = main_q[i]
            target_q.append(current_target)

        ax.plot(steps, main_q, label="Main Network Q(s,a)", alpha=0.8, linewidth=1)
        ax.plot(
            steps,
            target_q,
            label="Target Network Q(s,a)",
            alpha=0.8,
            linewidth=2,
            drawstyle="steps-post",
        )

        ax.set_title("Target Network Update Schedule")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Q-Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Epsilon-Greedy Exploration
        ax = axes[1, 1]

        episodes = np.arange(0, 1000)
        epsilon_decay = 0.995
        epsilon_min = 0.01
        epsilon_values = []

        epsilon = 1.0
        for episode in episodes:
            epsilon_values.append(epsilon)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        ax.plot(episodes, epsilon_values, linewidth=3, color="red")
        ax.fill_between(episodes, epsilon_values, alpha=0.3, color="red")

        ax.set_title("ε-Greedy Exploration Schedule")
        ax.set_xlabel("Training Episodes")
        ax.set_ylabel("Epsilon Value")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.show()

    def demonstrate_overestimation_bias(self):
        """Demonstrate the overestimation bias problem"""

        # Simulate true Q-values and noisy estimates
        np.random.seed(42)
        true_q_values = np.array([1.0, 2.0, 1.5, 0.8, 2.2])
        noise_std = 0.5
        num_estimates = 1000

        # Generate noisy estimates
        estimates = []
        max_estimates = []

        for _ in range(num_estimates):
            noisy_q = true_q_values + np.random.normal(0, noise_std, len(true_q_values))
            estimates.append(noisy_q)
            max_estimates.append(np.max(noisy_q))

        estimates = np.array(estimates)

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Distribution of max Q-values
        ax = axes[0]
        ax.hist(max_estimates, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        ax.axvline(
            np.max(true_q_values),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"True Max: {np.max(true_q_values):.2f}",
        )
        ax.axvline(
            np.mean(max_estimates),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Estimated Max: {np.mean(max_estimates):.2f}",
        )

        ax.set_title("Overestimation Bias in Max Q-Values")
        ax.set_xlabel("Max Q-Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Q-value distributions for each action
        ax = axes[1]
        positions = np.arange(len(true_q_values))

        violin_parts = ax.violinplot(
            [estimates[:, i] for i in range(len(true_q_values))],
            positions=positions,
            showmeans=True,
            showmedians=True,
        )

        # Plot true values
        ax.scatter(
            positions,
            true_q_values,
            color="red",
            s=100,
            zorder=10,
            label="True Q-Values",
            marker="D",
        )

        ax.set_title("Q-Value Distributions with Noise")
        ax.set_xlabel("Actions")
        ax.set_ylabel("Q-Values")
        ax.set_xticks(positions)
        ax.set_xticklabels([f"A{i}" for i in range(len(true_q_values))])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Calculate and print bias
        bias = np.mean(max_estimates) - np.max(true_q_values)
        print(f"Overestimation Bias: {bias:.3f}")
        print(f"True Maximum Q-Value: {np.max(true_q_values):.3f}")
        print(f"Average Estimated Maximum: {np.mean(max_estimates):.3f}")


class PerformanceAnalyzer:
    """
    Tools for analyzing DQN agent performance and behavior
    """

    @staticmethod
    def plot_learning_curves(agents_results: Dict[str, Dict], window_size: int = 10):
        """
        Plot learning curves for multiple agents

        Args:
            agents_results: Dictionary with agent names as keys and results as values
            window_size: Smoothing window size
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        colors = ["blue", "red", "green", "orange", "purple"]

        # Learning curves
        ax = axes[0, 0]
        for i, (agent_name, results) in enumerate(agents_results.items()):
            rewards = results.get("rewards", [])
            if rewards:
                smoothed = pd.Series(rewards).rolling(window_size).mean()
                ax.plot(
                    smoothed,
                    label=agent_name,
                    color=colors[i % len(colors)],
                    linewidth=2,
                )

        ax.set_title("Learning Curves Comparison")
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Episode Reward (Smoothed {window_size})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Final performance comparison
        ax = axes[0, 1]
        agent_names = list(agents_results.keys())
        final_perfs = []

        for agent_name in agent_names:
            rewards = agents_results[agent_name].get("rewards", [])
            if rewards:
                final_perf = np.mean(rewards[-20:])  # Last 20 episodes
                final_perfs.append(final_perf)
            else:
                final_perfs.append(0)

        bars = ax.bar(
            agent_names, final_perfs, alpha=0.7, color=colors[: len(agent_names)]
        )
        ax.set_title("Final Performance Comparison")
        ax.set_ylabel("Average Reward (Last 20 Episodes)")
        ax.set_xticklabels(agent_names, rotation=45)

        # Add value labels on bars
        for bar, perf in zip(bars, final_perfs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{perf:.1f}",
                ha="center",
                va="bottom",
            )

        ax.grid(True, alpha=0.3)

        # Loss curves
        ax = axes[1, 0]
        for i, (agent_name, results) in enumerate(agents_results.items()):
            losses = results.get("losses", [])
            if losses and len(losses) > 50:
                smoothed_loss = pd.Series(losses).rolling(50).mean()
                ax.plot(
                    smoothed_loss,
                    label=agent_name,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    alpha=0.7,
                )

        ax.set_title("Training Loss Comparison")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MSE Loss (Smoothed)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Epsilon decay
        ax = axes[1, 1]
        for i, (agent_name, results) in enumerate(agents_results.items()):
            epsilon_history = results.get("epsilon_history", [])
            if epsilon_history:
                ax.plot(
                    epsilon_history,
                    label=agent_name,
                    color=colors[i % len(colors)],
                    linewidth=2,
                )

        ax.set_title("Epsilon Decay Comparison")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Epsilon Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def analyze_q_value_distributions(agent, env, num_samples: int = 1000):
        """
        Analyze Q-value distributions across different states

        Args:
            agent: Trained DQN agent
            env: Gym environment
            num_samples: Number of state samples to analyze
        """
        print("=" * 60)
        print("Q-Value Distribution Analysis")
        print("=" * 60)

        # Collect states by sampling from environment
        states = []
        for _ in range(num_samples):
            state, _ = env.reset()
            states.append(state)

            # Take a few random steps to get diverse states
            for _ in range(np.random.randint(1, 10)):
                action = env.action_space.sample()
                state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
                states.append(state)

        states = np.array(states[:num_samples])

        # Get Q-values for all states
        q_values_all = []
        for state in states:
            q_vals = agent.get_q_values(state)
            q_values_all.append(q_vals)

        q_values_all = np.array(q_values_all)

        # Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Q-value distributions for each action
        ax = axes[0, 0]
        for i in range(agent.action_dim):
            ax.hist(
                q_values_all[:, i],
                bins=30,
                alpha=0.6,
                label=f"Action {i}",
                density=True,
            )

        ax.set_title("Q-Value Distributions by Action")
        ax.set_xlabel("Q-Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Average Q-values per action
        ax = axes[0, 1]
        mean_q_per_action = np.mean(q_values_all, axis=0)
        std_q_per_action = np.std(q_values_all, axis=0)

        bars = ax.bar(
            range(agent.action_dim),
            mean_q_per_action,
            yerr=std_q_per_action,
            alpha=0.7,
            capsize=5,
        )
        ax.set_title("Average Q-Values by Action")
        ax.set_xlabel("Action")
        ax.set_ylabel("Average Q-Value")
        ax.set_xticks(range(agent.action_dim))
        ax.grid(True, alpha=0.3)

        # Q-value ranges
        ax = axes[1, 0]
        q_ranges = np.ptp(q_values_all, axis=0)  # Peak-to-peak (max - min)
        ax.bar(range(agent.action_dim), q_ranges, alpha=0.7)
        ax.set_title("Q-Value Ranges by Action")
        ax.set_xlabel("Action")
        ax.set_ylabel("Q-Value Range (Max - Min)")
        ax.set_xticks(range(agent.action_dim))
        ax.grid(True, alpha=0.3)

        # State-wise Q-value statistics
        ax = axes[1, 1]
        max_q_per_state = np.max(q_values_all, axis=1)
        mean_q_per_state = np.mean(q_values_all, axis=1)

        ax.scatter(mean_q_per_state, max_q_per_state, alpha=0.6, s=10)
        ax.set_title("State-wise Q-Value Statistics")
        ax.set_xlabel("Mean Q-Value Across Actions")
        ax.set_ylabel("Max Q-Value")
        ax.grid(True, alpha=0.3)

        # Add diagonal line
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="y=x")

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\nQ-Value Statistics (across {num_samples} states):")
        print(
            f"Overall Q-value range: [{np.min(q_values_all):.3f}, {np.max(q_values_all):.3f}]"
        )
        print(
            f"Average Q-value: {np.mean(q_values_all):.3f} ± {np.std(q_values_all):.3f}"
        )

        for i in range(agent.action_dim):
            q_action = q_values_all[:, i]
            print(
                f"Action {i}: Mean={np.mean(q_action):.3f}, Std={np.std(q_action):.3f}, "
                f"Range=[{np.min(q_action):.3f}, {np.max(q_action):.3f}]"
            )

        # Return agent and analysis results
        analysis_results = {
            "q_values_all": q_values_all,
            "mean_q_per_action": mean_q_per_action,
            "std_q_per_action": std_q_per_action,
            "q_ranges": q_ranges,
            "max_q_per_state": max_q_per_state,
            "mean_q_per_state": mean_q_per_state,
            "num_samples": num_samples,
        }

        return agent, analysis_results

    @staticmethod
    def compare_policies(agents: Dict[str, Any], env, num_episodes: int = 10):
        """
        Compare policies of different agents

        Args:
            agents: Dictionary of agent names to agents
            env: Gym environment
            num_episodes: Number of episodes to evaluate each agent
        """
        print("=" * 60)
        print("Policy Comparison")
        print("=" * 60)

        results = {}

        for agent_name, agent in agents.items():
            print(f"\nEvaluating {agent_name}...")

            # Temporarily disable exploration
            original_epsilon = getattr(agent, "epsilon", 0.0)
            agent.epsilon = 0.0

            episode_rewards = []
            episode_lengths = []

            for episode in range(num_episodes):
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0

                while True:
                    action = agent.select_action(state)
                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    episode_length += 1

                    if terminated or truncated:
                        break

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

            # Restore original epsilon
            if hasattr(agent, "epsilon"):
                agent.epsilon = original_epsilon

            results[agent_name] = {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "mean_length": np.mean(episode_lengths),
                "std_length": np.std(episode_lengths),
                "rewards": episode_rewards,
                "lengths": episode_lengths,
            }

            print(
                f"  Mean Reward: {results[agent_name]['mean_reward']:.2f} ± {results[agent_name]['std_reward']:.2f}"
            )
            print(
                f"  Mean Length: {results[agent_name]['mean_length']:.1f} ± {results[agent_name]['std_length']:.1f}"
            )

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Performance comparison
        ax = axes[0]
        agent_names = list(results.keys())
        means = [results[name]["mean_reward"] for name in agent_names]
        stds = [results[name]["std_reward"] for name in agent_names]

        bars = ax.bar(agent_names, means, yerr=stds, alpha=0.7, capsize=5)
        ax.set_title("Policy Performance Comparison")
        ax.set_ylabel("Average Episode Reward")
        ax.set_xticklabels(agent_names, rotation=45)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
            )

        ax.grid(True, alpha=0.3)

        # Episode length comparison
        ax = axes[1]
        lengths = [results[name]["mean_length"] for name in agent_names]
        length_stds = [results[name]["std_length"] for name in agent_names]

        bars = ax.bar(agent_names, lengths, yerr=length_stds, alpha=0.7, capsize=5)
        ax.set_title("Episode Length Comparison")
        ax.set_ylabel("Average Episode Length")
        ax.set_xticklabels(agent_names, rotation=45)

        # Add value labels
        for bar, length in zip(bars, lengths):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{length:.1f}",
                ha="center",
                va="bottom",
            )

        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return results
