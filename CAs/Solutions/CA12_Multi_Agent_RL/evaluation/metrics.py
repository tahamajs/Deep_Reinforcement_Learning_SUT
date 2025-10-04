"""
Metrics for evaluating multi-agent RL performance.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class MultiAgentMetrics:
    """Comprehensive metrics for multi-agent RL evaluation."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.agent_rewards = defaultdict(list)
        self.coordination_scores = []
        self.efficiency_scores = []
        self.communication_rates = []
        self.convergence_episodes = []

        # Detailed tracking
        self.action_distributions = defaultdict(list)
        self.value_estimates = defaultdict(list)
        self.policy_entropies = defaultdict(list)

    def update(
        self,
        episode_reward: float,
        episode_length: int,
        agent_rewards: List[float],
        coordination_score: float,
        efficiency_score: float,
        communication_rate: float = 0.0,
        action_distributions: Optional[Dict[int, np.ndarray]] = None,
        value_estimates: Optional[Dict[int, float]] = None,
        policy_entropies: Optional[Dict[int, float]] = None,
    ):
        """Update metrics with new episode data."""

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.coordination_scores.append(coordination_score)
        self.efficiency_scores.append(efficiency_score)
        self.communication_rates.append(communication_rate)

        for i, reward in enumerate(agent_rewards):
            self.agent_rewards[i].append(reward)

        if action_distributions:
            for agent_id, dist in action_distributions.items():
                self.action_distributions[agent_id].append(dist)

        if value_estimates:
            for agent_id, value in value_estimates.items():
                self.value_estimates[agent_id].append(value)

        if policy_entropies:
            for agent_id, entropy in policy_entropies.items():
                self.policy_entropies[agent_id].append(entropy)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.episode_rewards:
            return {}

        stats = {
            "total_episodes": len(self.episode_rewards),
            "mean_episode_reward": np.mean(self.episode_rewards),
            "std_episode_reward": np.std(self.episode_rewards),
            "mean_episode_length": np.mean(self.episode_lengths),
            "mean_coordination_score": np.mean(self.coordination_scores),
            "mean_efficiency_score": np.mean(self.efficiency_scores),
            "mean_communication_rate": np.mean(self.communication_rates),
        }

        # Agent-specific stats
        for i in range(self.n_agents):
            if i in self.agent_rewards:
                stats[f"agent_{i}_mean_reward"] = np.mean(self.agent_rewards[i])
                stats[f"agent_{i}_std_reward"] = np.std(self.agent_rewards[i])

        return stats

    def plot_learning_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot learning curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.7)
        axes[0, 0].plot(self._smooth_curve(self.episode_rewards), linewidth=2)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)

        # Agent rewards
        for i in range(self.n_agents):
            if i in self.agent_rewards:
                axes[0, 1].plot(self.agent_rewards[i], alpha=0.7, label=f"Agent {i}")
        axes[0, 1].set_title("Individual Agent Rewards")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Coordination scores
        axes[0, 2].plot(self.coordination_scores, alpha=0.7)
        axes[0, 2].plot(self._smooth_curve(self.coordination_scores), linewidth=2)
        axes[0, 2].set_title("Coordination Scores")
        axes[0, 2].set_xlabel("Episode")
        axes[0, 2].set_ylabel("Coordination Score")
        axes[0, 2].grid(True)

        # Efficiency scores
        axes[1, 0].plot(self.efficiency_scores, alpha=0.7)
        axes[1, 0].plot(self._smooth_curve(self.efficiency_scores), linewidth=2)
        axes[1, 0].set_title("Efficiency Scores")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Efficiency Score")
        axes[1, 0].grid(True)

        # Communication rates
        if any(rate > 0 for rate in self.communication_rates):
            axes[1, 1].plot(self.communication_rates, alpha=0.7)
            axes[1, 1].plot(self._smooth_curve(self.communication_rates), linewidth=2)
            axes[1, 1].set_title("Communication Rates")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Communication Rate")
            axes[1, 1].grid(True)

        # Episode lengths
        axes[1, 2].plot(self.episode_lengths, alpha=0.7)
        axes[1, 2].plot(self._smooth_curve(self.episode_lengths), linewidth=2)
        axes[1, 2].set_title("Episode Lengths")
        axes[1, 2].set_xlabel("Episode")
        axes[1, 2].set_ylabel("Length")
        axes[1, 2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _smooth_curve(self, data: List[float], window: int = 100) -> List[float]:
        """Apply moving average smoothing."""
        if len(data) < window:
            return data

        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2)
            smoothed.append(np.mean(data[start:end]))

        return smoothed


def compute_coordination_score(
    actions: List[List[int]], coordination_type: str = "consensus"
) -> float:
    """Compute coordination score based on action patterns."""
    if not actions or not actions[0]:
        return 0.0

    actions = np.array(actions)
    n_agents, n_steps = actions.shape

    if coordination_type == "consensus":
        # Measure how often agents agree on actions
        consensus_score = 0.0
        for t in range(n_steps):
            unique_actions = len(np.unique(actions[:, t]))
            step_consensus = (n_agents - unique_actions + 1) / n_agents
            consensus_score += step_consensus
        return consensus_score / n_steps

    elif coordination_type == "diversity":
        # Measure action diversity (inverse of coordination)
        diversity_score = 0.0
        for t in range(n_steps):
            unique_actions = len(np.unique(actions[:, t]))
            diversity_score += unique_actions / n_agents
        return diversity_score / n_steps

    elif coordination_type == "sequence":
        # Measure sequential coordination patterns
        sequence_score = 0.0
        for agent in range(n_agents):
            for other_agent in range(agent + 1, n_agents):
                correlation = np.corrcoef(actions[agent], actions[other_agent])[0, 1]
                if not np.isnan(correlation):
                    sequence_score += abs(correlation)
        return sequence_score / (n_agents * (n_agents - 1) / 2)

    else:
        return 0.0


def compute_efficiency_score(
    rewards: List[float],
    max_possible_reward: float = 100.0,
    min_possible_reward: float = 0.0,
) -> float:
    """Compute efficiency score based on reward achievement."""
    if not rewards:
        return 0.0

    normalized_rewards = [
        (r - min_possible_reward) / (max_possible_reward - min_possible_reward)
        for r in rewards
    ]

    # Efficiency is the ratio of achieved reward to maximum possible
    return np.mean(normalized_rewards)


def compute_communication_efficiency(
    messages_sent: List[int], messages_received: List[int], total_possible: int
) -> float:
    """Compute communication efficiency metrics."""
    if not messages_sent or not messages_received:
        return 0.0

    sent_rate = np.mean(messages_sent) / total_possible if total_possible > 0 else 0.0
    received_rate = (
        np.mean(messages_received) / total_possible if total_possible > 0 else 0.0
    )

    # Communication efficiency is the harmonic mean of sent and received rates
    if sent_rate + received_rate == 0:
        return 0.0
    return 2 * sent_rate * received_rate / (sent_rate + received_rate)


def analyze_policy_convergence(
    metrics: MultiAgentMetrics, window_size: int = 100, threshold: float = 0.01
) -> Dict[str, Any]:
    """Analyze policy convergence patterns."""
    if len(metrics.episode_rewards) < window_size:
        return {"converged": False, "convergence_episode": -1}

    # Check if rewards have stabilized
    recent_rewards = metrics.episode_rewards[-window_size:]
    reward_variance = np.var(recent_rewards)

    converged = reward_variance < threshold
    convergence_episode = (
        len(metrics.episode_rewards) - window_size if converged else -1
    )

    return {
        "converged": converged,
        "convergence_episode": convergence_episode,
        "final_variance": reward_variance,
        "improvement_trend": np.mean(recent_rewards)
        - np.mean(metrics.episode_rewards[:window_size]),
    }


def compare_algorithm_performance(
    metrics_list: List[MultiAgentMetrics], algorithm_names: List[str]
) -> Dict[str, Any]:
    """Compare performance across multiple algorithms."""
    comparison = {}

    for i, (metrics, name) in enumerate(zip(metrics_list, algorithm_names)):
        stats = metrics.get_summary_stats()
        comparison[name] = {
            "mean_reward": stats.get("mean_episode_reward", 0.0),
            "std_reward": stats.get("std_episode_reward", 0.0),
            "mean_coordination": stats.get("mean_coordination_score", 0.0),
            "mean_efficiency": stats.get("mean_efficiency_score", 0.0),
            "convergence_analysis": analyze_policy_convergence(metrics),
        }

    return comparison
