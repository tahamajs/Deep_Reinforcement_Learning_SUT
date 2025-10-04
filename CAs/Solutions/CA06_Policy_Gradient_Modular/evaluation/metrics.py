"""
Evaluation metrics for policy gradient methods
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from collections import deque
import statistics


class PolicyGradientEvaluator:
    """Comprehensive evaluator for policy gradient algorithms"""

    def __init__(self, env_name: str, num_eval_episodes: int = 100):
        self.env_name = env_name
        self.num_eval_episodes = num_eval_episodes
        self.env = gym.make(env_name)

    def evaluate_agent(self, agent, render: bool = False) -> Dict[str, float]:
        """Evaluate agent performance"""
        episode_rewards = []
        episode_lengths = []

        for episode in range(self.num_eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                if hasattr(agent, "select_action"):
                    if "continuous" in self.env_name.lower():
                        action, _ = agent.select_action(state)
                    else:
                        action, _ = agent.select_action(state)
                else:
                    action = agent.get_action(state)

                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1

                if render and episode < 3:  # Render only first 3 episodes
                    self.env.render()

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        self.env.close()

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }


class PerformanceAnalyzer:
    """Analyze training performance and convergence"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.training_rewards = []
        self.training_losses = []
        self.training_lengths = []

    def add_episode(self, reward: float, loss: float = None, length: int = None):
        """Add episode data"""
        self.training_rewards.append(reward)
        if loss is not None:
            self.training_losses.append(loss)
        if length is not None:
            self.training_lengths.append(length)

    def get_smoothed_rewards(self) -> List[float]:
        """Get smoothed rewards using moving average"""
        if len(self.training_rewards) < self.window_size:
            return self.training_rewards

        smoothed = []
        for i in range(len(self.training_rewards)):
            start_idx = max(0, i - self.window_size + 1)
            smoothed.append(np.mean(self.training_rewards[start_idx : i + 1]))

        return smoothed

    def get_convergence_metrics(self) -> Dict[str, float]:
        """Calculate convergence metrics"""
        if len(self.training_rewards) < 2 * self.window_size:
            return {}

        # Recent performance
        recent_rewards = self.training_rewards[-self.window_size :]
        early_rewards = self.training_rewards[: self.window_size]

        # Performance improvement
        improvement = np.mean(recent_rewards) - np.mean(early_rewards)

        # Stability (coefficient of variation)
        stability = np.std(recent_rewards) / (np.mean(recent_rewards) + 1e-8)

        # Trend analysis
        if len(recent_rewards) >= 10:
            x = np.arange(len(recent_rewards))
            trend_slope = np.polyfit(x, recent_rewards, 1)[0]
        else:
            trend_slope = 0

        return {
            "improvement": improvement,
            "stability": stability,
            "trend_slope": trend_slope,
            "mean_recent_performance": np.mean(recent_rewards),
            "std_recent_performance": np.std(recent_rewards),
        }

    def has_converged(self, threshold: float = 0.01, min_episodes: int = 200) -> bool:
        """Check if training has converged"""
        if len(self.training_rewards) < min_episodes:
            return False

        recent_rewards = self.training_rewards[-self.window_size :]
        early_rewards = self.training_rewards[-2 * self.window_size : -self.window_size]

        if len(early_rewards) == 0:
            return False

        # Check if recent performance is stable
        recent_std = np.std(recent_rewards)
        recent_mean = np.mean(recent_rewards)

        # Check if improvement is minimal
        improvement = np.mean(recent_rewards) - np.mean(early_rewards)

        return (recent_std / (recent_mean + 1e-8)) < threshold and abs(
            improvement
        ) < threshold


class ConvergenceAnalyzer:
    """Analyze convergence patterns across different algorithms"""

    def __init__(self):
        self.algorithm_results = {}

    def add_algorithm_results(
        self,
        algorithm_name: str,
        rewards: List[float],
        losses: List[float] = None,
        lengths: List[float] = None,
    ):
        """Add results for an algorithm"""
        self.algorithm_results[algorithm_name] = {
            "rewards": rewards,
            "losses": losses or [],
            "lengths": lengths or [],
        }

    def compare_convergence(self) -> Dict[str, Dict]:
        """Compare convergence across algorithms"""
        comparison = {}

        for algo_name, results in self.algorithm_results.items():
            rewards = results["rewards"]

            if len(rewards) < 100:
                continue

            # Calculate convergence metrics
            window_size = min(50, len(rewards) // 4)

            # Early vs late performance
            early_perf = np.mean(rewards[:window_size])
            late_perf = np.mean(rewards[-window_size:])
            improvement = late_perf - early_perf

            # Stability
            late_rewards = rewards[-window_size:]
            stability = np.std(late_rewards) / (np.mean(late_rewards) + 1e-8)

            # Convergence speed (episodes to reach 80% of final performance)
            target_perf = early_perf + 0.8 * improvement
            convergence_episode = None

            for i, reward in enumerate(rewards):
                if reward >= target_perf:
                    convergence_episode = i
                    break

            comparison[algo_name] = {
                "early_performance": early_perf,
                "late_performance": late_perf,
                "improvement": improvement,
                "stability": stability,
                "convergence_episode": convergence_episode,
                "total_episodes": len(rewards),
            }

        return comparison

    def get_best_algorithm(self, metric: str = "late_performance") -> str:
        """Get the best performing algorithm based on metric"""
        comparison = self.compare_convergence()

        if not comparison:
            return None

        best_algo = max(comparison.keys(), key=lambda x: comparison[x][metric])
        return best_algo


def calculate_sample_efficiency(
    rewards: List[float], target_performance: float, window_size: int = 50
) -> Optional[int]:
    """Calculate sample efficiency - episodes to reach target performance"""
    if len(rewards) < window_size:
        return None

    # Smooth the rewards
    smoothed_rewards = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        smoothed_rewards.append(np.mean(rewards[start_idx : i + 1]))

    # Find first episode where smoothed reward exceeds target
    for i, reward in enumerate(smoothed_rewards):
        if reward >= target_performance:
            return i

    return None


def calculate_final_performance(
    rewards: List[float], window_size: int = 100
) -> Dict[str, float]:
    """Calculate final performance metrics"""
    if len(rewards) < window_size:
        window_size = len(rewards)

    final_rewards = rewards[-window_size:]

    return {
        "mean_final_performance": np.mean(final_rewards),
        "std_final_performance": np.std(final_rewards),
        "median_final_performance": np.median(final_rewards),
        "min_final_performance": np.min(final_rewards),
        "max_final_performance": np.max(final_rewards),
    }

