"""
Algorithm comparison utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from .metrics import MultiAgentMetrics, compare_algorithm_performance


class AlgorithmComparator:
    """Compare multiple multi-agent RL algorithms."""

    def __init__(self):
        self.results = {}
        self.statistical_tests = {}

    def add_algorithm(self, name: str, metrics: MultiAgentMetrics):
        """Add algorithm results for comparison."""
        self.results[name] = metrics

    def compare_performance(self, metric: str = "episode_rewards") -> Dict[str, Any]:
        """Compare algorithms on specified metric."""
        comparison_data = {}

        for name, metrics in self.results.items():
            if metric == "episode_rewards":
                comparison_data[name] = metrics.episode_rewards
            elif metric == "coordination_scores":
                comparison_data[name] = metrics.coordination_scores
            elif metric == "efficiency_scores":
                comparison_data[name] = metrics.efficiency_scores
            else:
                comparison_data[name] = getattr(metrics, metric, [])

        return comparison_data

    def statistical_significance_test(
        self, algorithm1: str, algorithm2: str, metric: str = "episode_rewards"
    ) -> Dict[str, float]:
        """Perform statistical significance test between two algorithms."""
        from scipy import stats

        if algorithm1 not in self.results or algorithm2 not in self.results:
            return {}

        data1 = getattr(self.results[algorithm1], metric, [])
        data2 = getattr(self.results[algorithm2], metric, [])

        if not data1 or not data2:
            return {}

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        cohens_d = (
            (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
        )

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
            "effect_size": (
                "large"
                if abs(cohens_d) > 0.8
                else "medium" if abs(cohens_d) > 0.5 else "small"
            ),
        }

    def plot_comparison(
        self,
        metrics: List[str] = [
            "episode_rewards",
            "coordination_scores",
            "efficiency_scores",
        ],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot comparison across multiple metrics."""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            comparison_data = self.compare_performance(metric)

            # Create box plot
            data_for_plot = []
            labels = []
            for name, data in comparison_data.items():
                data_for_plot.append(data)
                labels.append(name)

            if data_for_plot:
                axes[i].boxplot(data_for_plot, labels=labels)
                axes[i].set_title(f"{metric.replace('_', ' ').title()}")
                axes[i].set_ylabel("Value")
                axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def generate_report(self) -> str:
        """Generate comprehensive comparison report."""
        report = "Multi-Agent RL Algorithm Comparison Report\n"
        report += "=" * 50 + "\n\n"

        # Summary statistics
        for name, metrics in self.results.items():
            stats = metrics.get_summary_stats()
            report += f"{name}:\n"
            report += f"  Mean Episode Reward: {stats.get('mean_episode_reward', 0.0):.3f} ± {stats.get('std_episode_reward', 0.0):.3f}\n"
            report += f"  Mean Coordination: {stats.get('mean_coordination_score', 0.0):.3f}\n"
            report += (
                f"  Mean Efficiency: {stats.get('mean_efficiency_score', 0.0):.3f}\n"
            )
            report += f"  Total Episodes: {stats.get('total_episodes', 0)}\n\n"

        # Statistical comparisons
        algorithm_names = list(self.results.keys())
        if len(algorithm_names) >= 2:
            report += "Statistical Comparisons:\n"
            report += "-" * 25 + "\n"

            for i in range(len(algorithm_names)):
                for j in range(i + 1, len(algorithm_names)):
                    test_result = self.statistical_significance_test(
                        algorithm_names[i], algorithm_names[j]
                    )
                    if test_result:
                        report += f"{algorithm_names[i]} vs {algorithm_names[j]}:\n"
                        report += f"  p-value: {test_result['p_value']:.4f}\n"
                        report += f"  Significant: {test_result['significant']}\n"
                        report += f"  Effect Size: {test_result['effect_size']}\n\n"

        return report


class PerformanceAnalyzer:
    """Analyze performance patterns and trends."""

    def __init__(self, metrics: MultiAgentMetrics):
        self.metrics = metrics

    def analyze_learning_phases(self, window_size: int = 100) -> Dict[str, Any]:
        """Analyze different phases of learning."""
        if len(self.metrics.episode_rewards) < window_size * 3:
            return {"error": "Insufficient data for phase analysis"}

        total_episodes = len(self.metrics.episode_rewards)
        n_phases = min(3, total_episodes // window_size)

        phases = {}
        for i in range(n_phases):
            start = i * window_size
            end = min((i + 1) * window_size, total_episodes)

            phase_rewards = self.metrics.episode_rewards[start:end]
            phase_coordination = self.metrics.coordination_scores[start:end]

            phases[f"phase_{i+1}"] = {
                "episodes": (start, end),
                "mean_reward": np.mean(phase_rewards),
                "std_reward": np.std(phase_rewards),
                "mean_coordination": np.mean(phase_coordination),
                "improvement_rate": self._compute_improvement_rate(phase_rewards),
                "stability": 1.0 / (np.std(phase_rewards) + 1e-8),
            }

        return phases

    def _compute_improvement_rate(self, rewards: List[float]) -> float:
        """Compute rate of improvement in rewards."""
        if len(rewards) < 2:
            return 0.0

        # Linear regression slope
        x = np.arange(len(rewards))
        slope, _ = np.polyfit(x, rewards, 1)
        return slope

    def analyze_agent_contribution(self) -> Dict[str, Any]:
        """Analyze individual agent contributions."""
        agent_contributions = {}

        for agent_id, rewards in self.metrics.agent_rewards.items():
            agent_contributions[agent_id] = {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "reward_trend": self._compute_improvement_rate(rewards),
                "consistency": 1.0 / (np.std(rewards) + 1e-8),
                "contribution_to_team": (
                    np.mean(rewards) / np.mean(self.metrics.episode_rewards)
                    if self.metrics.episode_rewards
                    else 0.0
                ),
            }

        return agent_contributions

    def detect_performance_anomalies(
        self, threshold_std: float = 2.0
    ) -> Dict[str, Any]:
        """Detect anomalous performance episodes."""
        if not self.metrics.episode_rewards:
            return {}

        rewards = np.array(self.metrics.episode_rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        # Identify outliers
        z_scores = np.abs((rewards - mean_reward) / (std_reward + 1e-8))
        outliers = np.where(z_scores > threshold_std)[0]

        return {
            "outlier_episodes": outliers.tolist(),
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(rewards) * 100,
            "mean_z_score": np.mean(z_scores),
            "max_z_score": np.max(z_scores),
        }

    def compute_sample_efficiency(
        self,
        target_performance: float = 0.8,
        performance_metric: str = "episode_rewards",
    ) -> Dict[str, Any]:
        """Compute sample efficiency metrics."""
        data = getattr(self.metrics, performance_metric, [])
        if not data:
            return {}

        # Find episodes where performance first exceeds target
        max_performance = np.max(data)
        target_value = max_performance * target_performance

        episodes_to_target = None
        for i, value in enumerate(data):
            if value >= target_value:
                episodes_to_target = i + 1
                break

        return {
            "episodes_to_target": episodes_to_target,
            "target_performance": target_value,
            "max_performance": max_performance,
            "final_performance": data[-1] if data else 0.0,
            "sample_efficiency": (
                1.0 / (episodes_to_target + 1e-8) if episodes_to_target else 0.0
            ),
        }
