"""
Model-Based RL Evaluator
========================

Comprehensive evaluation framework for model-based reinforcement learning methods.
Provides standardized evaluation protocols and metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from typing import Dict, List, Tuple, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import TabularModel, NeuralModel
from environments.environments import SimpleGridWorld


class ModelBasedEvaluator:
    """Comprehensive evaluator for model-based RL methods"""

    def __init__(self, env, num_runs=5, seed=42):
        self.env = env
        self.num_runs = num_runs
        self.seed = seed
        self.results = {}

    def evaluate_agent(
        self,
        agent_class,
        agent_kwargs,
        num_episodes=100,
        max_steps=200,
        method_name="Unknown",
    ):
        """Evaluate an agent over multiple runs"""

        print(f"\n🔍 Evaluating {method_name}...")

        run_results = []

        for run in range(self.num_runs):
            # Set seed for reproducibility
            np.random.seed(self.seed + run)

            # Create fresh agent for each run
            agent = agent_class(**agent_kwargs)

            episode_rewards = []
            episode_lengths = []
            training_times = []

            start_time = time.time()

            for episode in range(num_episodes):
                episode_start = time.time()

                reward, length = agent.train_episode(self.env, max_steps=max_steps)

                episode_time = time.time() - episode_start

                episode_rewards.append(reward)
                episode_lengths.append(length)
                training_times.append(episode_time)

                if (episode + 1) % 20 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(
                        f"  Run {run+1}, Episode {episode+1}: Avg Reward = {avg_reward:.3f}"
                    )

            total_time = time.time() - start_time

            # Collect statistics
            run_stats = {
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "training_times": training_times,
                "total_time": total_time,
                "final_performance": np.mean(episode_rewards[-10:]),
                "learning_efficiency": self._calculate_learning_efficiency(
                    episode_rewards
                ),
                "sample_efficiency": self._calculate_sample_efficiency(episode_rewards),
                "stability": self._calculate_stability(episode_rewards),
            }

            # Add agent-specific statistics if available
            if hasattr(agent, "get_statistics"):
                run_stats.update(agent.get_statistics())

            run_results.append(run_stats)

        # Aggregate results across runs
        aggregated_results = self._aggregate_results(run_results)

        self.results[method_name] = aggregated_results

        # Print summary
        self._print_evaluation_summary(method_name, aggregated_results)

        return aggregated_results

    def _calculate_learning_efficiency(self, rewards):
        """Calculate learning efficiency (area under learning curve)"""
        return np.sum(rewards) / len(rewards)

    def _calculate_sample_efficiency(self, rewards):
        """Calculate sample efficiency (episodes to reach 80% of final performance)"""
        final_perf = np.mean(rewards[-10:])
        target_perf = 0.8 * final_perf

        for i, reward in enumerate(rewards):
            if np.mean(rewards[max(0, i - 9) : i + 1]) >= target_perf:
                return i + 1

        return len(rewards)

    def _calculate_stability(self, rewards):
        """Calculate stability (inverse of variance in final performance)"""
        final_rewards = rewards[-20:]
        return 1.0 / (np.var(final_rewards) + 1e-8)

    def _aggregate_results(self, run_results):
        """Aggregate results across multiple runs"""
        metrics = [
            "final_performance",
            "learning_efficiency",
            "sample_efficiency",
            "stability",
        ]

        aggregated = {}

        for metric in metrics:
            values = [run[metric] for run in run_results]
            aggregated[f"avg_{metric}"] = np.mean(values)
            aggregated[f"std_{metric}"] = np.std(values)

        # Aggregate episode rewards
        all_rewards = [run["episode_rewards"] for run in run_results]
        min_length = min(len(rewards) for rewards in all_rewards)
        truncated_rewards = [rewards[:min_length] for rewards in all_rewards]
        aggregated["avg_episode_rewards"] = np.mean(truncated_rewards, axis=0)
        aggregated["std_episode_rewards"] = np.std(truncated_rewards, axis=0)

        # Aggregate episode lengths
        all_lengths = [run["episode_lengths"] for run in run_results]
        truncated_lengths = [lengths[:min_length] for lengths in all_lengths]
        aggregated["avg_episode_lengths"] = np.mean(truncated_lengths, axis=0)
        aggregated["std_episode_lengths"] = np.std(truncated_lengths, axis=0)

        # Store individual run results
        aggregated["individual_runs"] = run_results

        return aggregated

    def _print_evaluation_summary(self, method_name, results):
        """Print evaluation summary"""
        print(f"\n📊 {method_name} Evaluation Summary:")
        print(
            f"  Final Performance: {results['avg_final_performance']:.3f} ± {results['std_final_performance']:.3f}"
        )
        print(
            f"  Learning Efficiency: {results['avg_learning_efficiency']:.3f} ± {results['std_learning_efficiency']:.3f}"
        )
        print(
            f"  Sample Efficiency: {results['avg_sample_efficiency']:.1f} ± {results['std_sample_efficiency']:.1f} episodes"
        )
        print(
            f"  Stability: {results['avg_stability']:.3f} ± {results['std_stability']:.3f}"
        )

    def compare_methods(self, save_path="visualizations"):
        """Compare all evaluated methods"""
        if not self.results:
            print("No results to compare. Run evaluations first.")
            return

        os.makedirs(save_path, exist_ok=True)

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model-Based RL Methods Comparison", fontsize=16)

        # Performance comparison
        methods = list(self.results.keys())
        performances = [
            self.results[method]["avg_final_performance"] for method in methods
        ]
        errors = [self.results[method]["std_final_performance"] for method in methods]

        axes[0, 0].bar(methods, performances, yerr=errors, capsize=5, alpha=0.7)
        axes[0, 0].set_title("Final Performance Comparison")
        axes[0, 0].set_ylabel("Average Episode Reward")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Learning efficiency comparison
        efficiencies = [
            self.results[method]["avg_learning_efficiency"] for method in methods
        ]
        eff_errors = [
            self.results[method]["std_learning_efficiency"] for method in methods
        ]

        axes[0, 1].bar(
            methods, efficiencies, yerr=eff_errors, capsize=5, alpha=0.7, color="green"
        )
        axes[0, 1].set_title("Learning Efficiency Comparison")
        axes[0, 1].set_ylabel("Average Reward over Episodes")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Sample efficiency comparison
        sample_effs = [
            self.results[method]["avg_sample_efficiency"] for method in methods
        ]
        sample_errors = [
            self.results[method]["std_sample_efficiency"] for method in methods
        ]

        axes[1, 0].bar(
            methods,
            sample_effs,
            yerr=sample_errors,
            capsize=5,
            alpha=0.7,
            color="orange",
        )
        axes[1, 0].set_title("Sample Efficiency Comparison")
        axes[1, 0].set_ylabel("Episodes to 80% Performance")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Learning curves
        for method in methods:
            rewards = self.results[method]["avg_episode_rewards"]
            smoothed = pd.Series(rewards).rolling(window=10).mean()
            axes[1, 1].plot(smoothed, label=method, linewidth=2)

        axes[1, 1].set_title("Learning Curves")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Episode Reward (Smoothed)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path}/method_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Print ranking
        self._print_method_ranking()

    def _print_method_ranking(self):
        """Print method ranking based on performance"""
        print(f"\n🏆 Method Ranking:")
        print("=" * 40)

        # Sort by final performance
        sorted_methods = sorted(
            self.results.items(),
            key=lambda x: x[1]["avg_final_performance"],
            reverse=True,
        )

        for i, (method, results) in enumerate(sorted_methods, 1):
            perf = results["avg_final_performance"]
            std = results["std_final_performance"]
            eff = results["avg_learning_efficiency"]
            sample_eff = results["avg_sample_efficiency"]

            print(f"{i}. {method}")
            print(f"   Performance: {perf:.3f} ± {std:.3f}")
            print(f"   Efficiency: {eff:.3f}")
            print(f"   Sample Efficiency: {sample_eff:.1f} episodes")
            print()

    def save_results(self, filepath="results/evaluation_results.pkl"):
        """Save evaluation results"""
        import pickle

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.results, f)

        print(f"Results saved to {filepath}")

    def load_results(self, filepath="results/evaluation_results.pkl"):
        """Load evaluation results"""
        import pickle

        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                self.results = pickle.load(f)
            print(f"Results loaded from {filepath}")
        else:
            print(f"No results file found at {filepath}")


def demonstrate_evaluation():
    """Demonstrate the evaluation framework"""

    print("Model-Based RL Evaluation Framework")
    print("=" * 40)

    # Create environment
    env = SimpleGridWorld(size=5)

    # Create evaluator
    evaluator = ModelBasedEvaluator(env, num_runs=3)

    # Import agent classes
    from agents.dyna_q import DynaQAgent

    # Evaluate different Dyna-Q configurations
    print("\n🔍 Running evaluations...")

    # Q-Learning baseline
    evaluator.evaluate_agent(
        DynaQAgent,
        {"num_states": 25, "num_actions": 4, "planning_steps": 0},
        num_episodes=50,
        method_name="Q-Learning",
    )

    # Dyna-Q with planning
    evaluator.evaluate_agent(
        DynaQAgent,
        {"num_states": 25, "num_actions": 4, "planning_steps": 10},
        num_episodes=50,
        method_name="Dyna-Q (n=10)",
    )

    # Dyna-Q with more planning
    evaluator.evaluate_agent(
        DynaQAgent,
        {"num_states": 25, "num_actions": 4, "planning_steps": 50},
        num_episodes=50,
        method_name="Dyna-Q (n=50)",
    )

    # Compare methods
    evaluator.compare_methods()

    # Save results
    evaluator.save_results()

    print("\n✅ Evaluation demonstration complete!")


if __name__ == "__main__":
    demonstrate_evaluation()
