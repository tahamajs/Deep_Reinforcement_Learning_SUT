"""
Evaluation utilities for CA07 DQN experiments
=============================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import gymnasium as gym
import json
import os
from collections import defaultdict


class DQNEvaluator:
    """Evaluation tools for DQN agents"""

    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def evaluate_agent(
        self,
        agent,
        env: gym.Env,
        num_episodes: int = 100,
        max_steps: int = 1000,
        epsilon: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluate agent performance"""
        rewards = []
        lengths = []
        q_values_history = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_q_values = []

            while episode_length < max_steps:
                # Get Q-values for analysis
                q_values = agent.get_q_values(state)
                episode_q_values.append(q_values)

                # Select action
                action = agent.select_action(state, epsilon=epsilon)

                # Take step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1
                state = next_state

                if done:
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)
            q_values_history.append(episode_q_values)

        # Calculate statistics
        stats = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "median_reward": np.median(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "success_rate": np.mean(np.array(rewards) >= 200),  # For CartPole
            "episode_rewards": rewards,
            "episode_lengths": lengths,
            "q_values_history": q_values_history,
        }

        return stats

    def compare_agents(
        self, agents: Dict[str, Any], env: gym.Env, num_episodes: int = 50
    ) -> Dict[str, Any]:
        """Compare multiple agents"""
        results = {}

        for agent_name, agent in agents.items():
            print(f"Evaluating {agent_name}...")
            stats = self.evaluate_agent(agent, env, num_episodes)
            results[agent_name] = stats

        # Create comparison plots
        self._plot_comparison(results)

        return results

    def _plot_comparison(self, results: Dict[str, Dict]):
        """Plot comparison of agent performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        agent_names = list(results.keys())

        # Reward comparison
        ax1 = axes[0, 0]
        mean_rewards = [results[name]["mean_reward"] for name in agent_names]
        std_rewards = [results[name]["std_reward"] for name in agent_names]

        bars = ax1.bar(
            agent_names,
            mean_rewards,
            yerr=std_rewards,
            alpha=0.7,
            capsize=5,
            color=["blue", "green", "red", "purple"],
        )
        ax1.set_title("Mean Reward Comparison")
        ax1.set_ylabel("Mean Reward")
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, mean, std in zip(bars, mean_rewards, std_rewards):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 1,
                f"{mean:.1f}±{std:.1f}",
                ha="center",
                va="bottom",
            )

        # Success rate comparison
        ax2 = axes[0, 1]
        success_rates = [results[name]["success_rate"] for name in agent_names]
        bars = ax2.bar(agent_names, success_rates, alpha=0.7, color="green")
        ax2.set_title("Success Rate Comparison")
        ax2.set_ylabel("Success Rate")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, rate in zip(bars, success_rates):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rate:.2f}",
                ha="center",
                va="bottom",
            )

        # Episode length comparison
        ax3 = axes[1, 0]
        mean_lengths = [results[name]["mean_length"] for name in agent_names]
        std_lengths = [results[name]["std_length"] for name in agent_names]

        bars = ax3.bar(
            agent_names,
            mean_lengths,
            yerr=std_lengths,
            alpha=0.7,
            capsize=5,
            color="orange",
        )
        ax3.set_title("Mean Episode Length Comparison")
        ax3.set_ylabel("Mean Length")
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for bar, mean, std in zip(bars, mean_lengths, std_lengths):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 1,
                f"{mean:.1f}±{std:.1f}",
                ha="center",
                va="bottom",
            )

        # Reward distribution
        ax4 = axes[1, 1]
        for i, (name, result) in enumerate(results.items()):
            rewards = result["episode_rewards"]
            ax4.hist(rewards, alpha=0.6, label=name, bins=20)
        ax4.set_title("Reward Distribution")
        ax4.set_xlabel("Reward")
        ax4.set_ylabel("Frequency")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, "agent_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def analyze_q_values(
        self, agent, env: gym.Env, num_states: int = 1000
    ) -> Dict[str, Any]:
        """Analyze Q-value patterns"""
        q_values_list = []
        states_list = []

        state, _ = env.reset()
        for _ in range(num_states):
            q_values = agent.get_q_values(state)
            q_values_list.append(q_values)
            states_list.append(state.copy())

            action = agent.select_action(state, epsilon=0.0)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                state, _ = env.reset()
            else:
                state = next_state

        q_values_array = np.array(q_values_list)

        # Calculate Q-value statistics
        q_stats = {
            "mean_q_values": np.mean(q_values_array, axis=0),
            "std_q_values": np.std(q_values_array, axis=0),
            "max_q_values": np.max(q_values_array, axis=0),
            "min_q_values": np.min(q_values_array, axis=0),
            "q_value_range": np.max(q_values_array) - np.min(q_values_array),
            "action_preferences": np.argmax(q_values_array, axis=1),
            "q_value_correlations": np.corrcoef(q_values_array.T),
        }

        # Plot Q-value analysis
        self._plot_q_value_analysis(q_values_array, q_stats)

        return q_stats

    def _plot_q_value_analysis(
        self, q_values_array: np.ndarray, q_stats: Dict[str, Any]
    ):
        """Plot Q-value analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Q-value distribution by action
        ax1 = axes[0, 0]
        for action in range(q_values_array.shape[1]):
            ax1.hist(
                q_values_array[:, action], alpha=0.6, label=f"Action {action}", bins=30
            )
        ax1.set_title("Q-value Distribution by Action")
        ax1.set_xlabel("Q-value")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-value correlation matrix
        ax2 = axes[0, 1]
        correlation_matrix = q_stats["q_value_correlations"]
        im = ax2.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax2.set_title("Q-value Correlation Matrix")
        ax2.set_xlabel("Action")
        ax2.set_ylabel("Action")

        # Add colorbar
        plt.colorbar(im, ax=ax2)

        # Add correlation values
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                ax2.text(
                    j,
                    i,
                    f"{correlation_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )

        # Action preferences
        ax3 = axes[1, 0]
        action_counts = np.bincount(q_stats["action_preferences"])
        ax3.bar(range(len(action_counts)), action_counts, alpha=0.7, color="green")
        ax3.set_title("Action Selection Frequency")
        ax3.set_xlabel("Action")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)

        # Q-value statistics
        ax4 = axes[1, 1]
        actions = range(q_values_array.shape[1])
        mean_q = q_stats["mean_q_values"]
        std_q = q_stats["std_q_values"]

        ax4.bar(actions, mean_q, yerr=std_q, alpha=0.7, capsize=5, color="purple")
        ax4.set_title("Mean Q-values by Action")
        ax4.set_xlabel("Action")
        ax4.set_ylabel("Mean Q-value")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, "q_value_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def save_results(
        self, results: Dict[str, Any], filename: str = "evaluation_results.json"
    ):
        """Save evaluation results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        serializable_results[key][sub_key] = sub_value.tolist()
                    else:
                        serializable_results[key][sub_key] = sub_value
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value

        with open(os.path.join(self.save_dir, filename), "w") as f:
            json.dump(serializable_results, f, indent=2)


def evaluate_training_progress(
    agent, env: gym.Env, num_episodes: int = 10
) -> Dict[str, float]:
    """Quick evaluation of training progress"""
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

            if done:
                break

        rewards.append(episode_reward)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "success_rate": np.mean(np.array(rewards) >= 200),
    }


def benchmark_agent(
    agent_class, env_name: str, num_runs: int = 5, episodes: int = 200
) -> Dict[str, Any]:
    """Benchmark agent across multiple runs"""
    results = []

    for run in range(num_runs):
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = agent_class(state_dim=state_dim, action_dim=action_dim)

        run_rewards = []
        for episode in range(episodes):
            reward, _ = agent.train_episode(env, max_steps=500)
            run_rewards.append(reward)

        results.append(run_rewards)
        env.close()

    # Calculate statistics across runs
    results_array = np.array(results)

    return {
        "mean_scores": np.mean(results_array, axis=0),
        "std_scores": np.std(results_array, axis=0),
        "final_mean": np.mean(results_array[:, -50:]),
        "final_std": np.std(results_array[:, -50:]),
        "individual_runs": results,
        "convergence_episode": find_convergence_episode(np.mean(results_array, axis=0)),
    }


def find_convergence_episode(
    scores: List[float], target: float = 180, window: int = 20
) -> int:
    """Find episode where agent converges to target score"""
    smoothed = np.convolve(scores, np.ones(window) / window, mode="valid")
    converged_idx = np.where(smoothed >= target)[0]
    return converged_idx[0] if len(converged_idx) > 0 else len(smoothed)
