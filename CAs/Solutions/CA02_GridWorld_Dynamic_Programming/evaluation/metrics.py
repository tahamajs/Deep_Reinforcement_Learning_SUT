"""
Performance Metrics and Evaluation Functions

This module provides comprehensive evaluation metrics for RL algorithms
including convergence analysis, performance comparison, and statistical measures.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import time

from environments.environments import GridWorld
from agents.policies import Policy
from agents.algorithms import (
    policy_evaluation,
    policy_iteration,
    value_iteration,
    q_learning,
)


def evaluate_policy_performance(
    env: GridWorld, policy: Policy, gamma: float = 0.9, num_episodes: int = 100
) -> Dict:
    """
    Evaluate policy performance using multiple metrics.

    Args:
        env: GridWorld environment
        policy: Policy to evaluate
        gamma: Discount factor
        num_episodes: Number of episodes for evaluation

    Returns:
        Dictionary containing performance metrics
    """
    start_time = time.time()

    # Compute value function
    values = policy_evaluation(env, policy, gamma)

    # Compute performance metrics
    start_value = values[env.start_state]
    goal_value = values[env.goal_state]

    # Calculate value variance (measure of consistency)
    non_terminal_values = [
        v for s, v in values.items() if s not in env.obstacles and s != env.goal_state
    ]
    value_variance = np.var(non_terminal_values) if non_terminal_values else 0

    # Calculate convergence time
    evaluation_time = time.time() - start_time

    # Simulate episodes to get empirical performance
    episode_rewards = []
    episode_lengths = []

    for _ in range(num_episodes):
        state = env.start_state
        total_reward = 0
        steps = 0
        max_steps = 100

        while steps < max_steps:
            if state == env.goal_state:
                break
            if state in env.obstacles:
                break

            action = policy.get_action(state)
            if action is None:
                break

            # Get transition
            transitions = env.P[state][action]
            prob, next_state, reward = transitions[0]

            total_reward += reward
            state = next_state
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    return {
        "start_value": start_value,
        "goal_value": goal_value,
        "value_variance": value_variance,
        "evaluation_time": evaluation_time,
        "mean_episode_reward": np.mean(episode_rewards),
        "std_episode_reward": np.std(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "success_rate": np.mean([r > 0 for r in episode_rewards]),
        "values": values,
    }


def compare_algorithm_convergence(env: GridWorld, gamma: float = 0.9) -> Dict:
    """
    Compare convergence properties of different algorithms.

    Args:
        env: GridWorld environment
        gamma: Discount factor

    Returns:
        Dictionary with convergence metrics for each algorithm
    """
    results = {}

    # Policy Iteration
    start_time = time.time()
    pi_policy, pi_values, pi_history = policy_iteration(env, gamma)
    pi_time = time.time() - start_time

    results["policy_iteration"] = {
        "iterations": len(pi_history),
        "convergence_time": pi_time,
        "final_value": pi_values[env.start_state],
        "history": pi_history,
    }

    # Value Iteration
    start_time = time.time()
    vi_values, vi_policy, vi_history = value_iteration(env, gamma)
    vi_time = time.time() - start_time

    results["value_iteration"] = {
        "iterations": len(vi_history),
        "convergence_time": vi_time,
        "final_value": vi_values[env.start_state],
        "history": vi_history,
    }

    # Q-Learning
    start_time = time.time()
    Q, episode_rewards = q_learning(env, num_episodes=1000, gamma=gamma)
    ql_time = time.time() - start_time

    # Calculate Q-learning convergence (when moving average stabilizes)
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
        convergence_episode = None
        for i in range(window, len(moving_avg)):
            if abs(moving_avg.iloc[i] - moving_avg.iloc[i - 1]) < 0.01:
                convergence_episode = i
                break
    else:
        convergence_episode = len(episode_rewards)

    results["q_learning"] = {
        "episodes": len(episode_rewards),
        "convergence_episode": convergence_episode,
        "convergence_time": ql_time,
        "final_reward": (
            np.mean(episode_rewards[-100:])
            if len(episode_rewards) >= 100
            else np.mean(episode_rewards)
        ),
        "episode_rewards": episode_rewards,
    }

    return results


def analyze_learning_efficiency(
    env: GridWorld, gamma_values: List[float] = [0.1, 0.5, 0.9, 0.99]
) -> Dict:
    """
    Analyze learning efficiency across different discount factors.

    Args:
        env: GridWorld environment
        gamma_values: List of discount factors to test

    Returns:
        Dictionary with efficiency metrics for each gamma
    """
    results = {}

    for gamma in gamma_values:
        gamma_results = {}

        # Policy Iteration efficiency
        start_time = time.time()
        pi_policy, pi_values, pi_history = policy_iteration(env, gamma)
        pi_time = time.time() - start_time

        gamma_results["policy_iteration"] = {
            "iterations": len(pi_history),
            "time": pi_time,
            "value": pi_values[env.start_state],
        }

        # Value Iteration efficiency
        start_time = time.time()
        vi_values, vi_policy, vi_history = value_iteration(env, gamma)
        vi_time = time.time() - start_time

        gamma_results["value_iteration"] = {
            "iterations": len(vi_history),
            "time": vi_time,
            "value": vi_values[env.start_state],
        }

        # Q-Learning efficiency (fewer episodes for efficiency test)
        start_time = time.time()
        Q, episode_rewards = q_learning(env, num_episodes=500, gamma=gamma)
        ql_time = time.time() - start_time

        gamma_results["q_learning"] = {
            "episodes": len(episode_rewards),
            "time": ql_time,
            "final_reward": (
                np.mean(episode_rewards[-50:])
                if len(episode_rewards) >= 50
                else np.mean(episode_rewards)
            ),
        }

        results[gamma] = gamma_results

    return results


def compute_performance_statistics(results: Dict) -> Dict:
    """
    Compute comprehensive performance statistics from evaluation results.

    Args:
        results: Dictionary containing evaluation results

    Returns:
        Dictionary with statistical summaries
    """
    stats = {}

    # Extract metrics for statistical analysis
    algorithms = ["policy_iteration", "value_iteration", "q_learning"]

    for alg in algorithms:
        if alg in results:
            alg_data = results[alg]

            if alg == "q_learning":
                # Q-learning specific metrics
                stats[alg] = {
                    "mean_final_reward": (
                        np.mean(alg_data["final_reward"])
                        if isinstance(alg_data["final_reward"], list)
                        else alg_data["final_reward"]
                    ),
                    "convergence_episodes": alg_data["convergence_episode"],
                    "total_time": alg_data["convergence_time"],
                }
            else:
                # Dynamic programming metrics
                stats[alg] = {
                    "iterations": alg_data["iterations"],
                    "convergence_time": alg_data["convergence_time"],
                    "final_value": alg_data["final_value"],
                }

    # Cross-algorithm comparisons
    if "policy_iteration" in stats and "value_iteration" in stats:
        stats["comparison"] = {
            "pi_vs_vi_iterations": stats["policy_iteration"]["iterations"]
            - stats["value_iteration"]["iterations"],
            "pi_vs_vi_time": stats["policy_iteration"]["convergence_time"]
            - stats["value_iteration"]["convergence_time"],
            "value_difference": abs(
                stats["policy_iteration"]["final_value"]
                - stats["value_iteration"]["final_value"]
            ),
        }

    return stats


def plot_performance_comparison(results: Dict, save_path: Optional[str] = None):
    """
    Create comprehensive performance comparison plots.

    Args:
        results: Evaluation results dictionary
        save_path: Optional path to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Algorithm convergence comparison
    ax1 = axes[0, 0]
    algorithms = []
    iterations = []
    times = []

    for alg, data in results.items():
        if alg in ["policy_iteration", "value_iteration"]:
            algorithms.append(alg.replace("_", " ").title())
            iterations.append(data["iterations"])
            times.append(data["convergence_time"])

    x = np.arange(len(algorithms))
    width = 0.35

    ax1.bar(x - width / 2, iterations, width, label="Iterations", alpha=0.8)
    ax1_twin = ax1.twinx()
    ax1_twin.bar(
        x + width / 2, times, width, label="Time (s)", alpha=0.8, color="orange"
    )

    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Iterations", color="blue")
    ax1_twin.set_ylabel("Time (s)", color="orange")
    ax1.set_title("Convergence Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)

    # Q-learning learning curve
    ax2 = axes[0, 1]
    if "q_learning" in results:
        episode_rewards = results["q_learning"]["episode_rewards"]
        episodes = range(1, len(episode_rewards) + 1)

        ax2.plot(
            episodes, episode_rewards, alpha=0.3, color="blue", label="Episode Reward"
        )

        # Moving average
        window = 50
        if len(episode_rewards) >= window:
            moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
            ax2.plot(
                episodes,
                moving_avg,
                color="red",
                linewidth=2,
                label=f"Moving Avg ({window})",
            )

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Reward")
        ax2.set_title("Q-Learning Progress")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Value function comparison
    ax3 = axes[1, 0]
    if "policy_iteration" in results and "value_iteration" in results:
        pi_value = results["policy_iteration"]["final_value"]
        vi_value = results["value_iteration"]["final_value"]

        algorithms = ["Policy Iteration", "Value Iteration"]
        values = [pi_value, vi_value]

        bars = ax3.bar(algorithms, values, color=["blue", "green"], alpha=0.7)
        ax3.set_ylabel("Start State Value")
        ax3.set_title("Final Value Comparison")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
            )

    # Performance summary
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create summary text
    summary_text = "Performance Summary:\n\n"

    if "policy_iteration" in results:
        pi_data = results["policy_iteration"]
        summary_text += f"Policy Iteration:\n"
        summary_text += f"  Iterations: {pi_data['iterations']}\n"
        summary_text += f"  Time: {pi_data['convergence_time']:.3f}s\n"
        summary_text += f"  Value: {pi_data['final_value']:.3f}\n\n"

    if "value_iteration" in results:
        vi_data = results["value_iteration"]
        summary_text += f"Value Iteration:\n"
        summary_text += f"  Iterations: {vi_data['iterations']}\n"
        summary_text += f"  Time: {vi_data['convergence_time']:.3f}s\n"
        summary_text += f"  Value: {vi_data['final_value']:.3f}\n\n"

    if "q_learning" in results:
        ql_data = results["q_learning"]
        summary_text += f"Q-Learning:\n"
        summary_text += f"  Episodes: {ql_data['episodes']}\n"
        summary_text += f"  Convergence: Episode {ql_data['convergence_episode']}\n"
        summary_text += f"  Final Reward: {ql_data['final_reward']:.3f}\n"

    ax4.text(
        0.1,
        0.9,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # Example usage
    from environments.environments import GridWorld
    from agents.policies import RandomPolicy

    env = GridWorld()

    # Run comprehensive evaluation
    print("Running comprehensive algorithm evaluation...")
    results = compare_algorithm_convergence(env)

    # Plot results
    plot_performance_comparison(results)

    print("Evaluation completed!")
