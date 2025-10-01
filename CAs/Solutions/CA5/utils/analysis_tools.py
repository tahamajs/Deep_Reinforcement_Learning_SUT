"""
DQN Analysis and Visualization Tools
===================================

This module provides comprehensive analysis and visualization tools
for DQN variants including performance comparison, convergence analysis,
hyperparameter sensitivity, and algorithmic insights.

Key Features:
- Performance comparison across DQN variants
- Convergence analysis and stability metrics
- Hyperparameter sensitivity analysis
- Algorithmic behavior visualization
- Statistical significance testing
- Learning dynamics analysis

Author: CA5 Implementation
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import uniform_filter1d
import pandas as pd

# Import agents - will be imported in notebook
try:
    from agents.dqn_base import DQNAgent
    from agents.double_dqn import DoubleDQNAgent
    from agents.dueling_dqn import DuelingDQNAgent
    from agents.prioritized_replay import PrioritizedDQNAgent
    from agents.rainbow_dqn import RainbowDQNAgent
except ImportError:
    # Fallback for when running as standalone module
    DQNAgent = None
    DoubleDQNAgent = None
    DuelingDQNAgent = None
    PrioritizedDQNAgent = None
    RainbowDQNAgent = None
import warnings

warnings.filterwarnings("ignore")


plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class DQNComparator:
    """Comprehensive comparison of DQN variants"""

    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.agents = {}
        self.results = {}

    def add_agent(self, name, agent_class, **kwargs):
        """Add agent for comparison"""
        self.agents[name] = {"class": agent_class, "kwargs": kwargs}

    def run_comparison(self, num_episodes=500, num_runs=3, save_results=True):
        """Run comprehensive comparison across all agents"""
        print(
            f"Running DQN Comparison: {len(self.agents)} agents, {num_runs} runs, {num_episodes} episodes each"
        )
        print("=" * 80)

        all_results = {}

        for agent_name, agent_config in self.agents.items():
            print(f"\\nTraining {agent_name}...")
            agent_scores = []

            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}")

                agent = agent_config["class"](
                    self.state_size, self.action_size, **agent_config["kwargs"]
                )

                scores, training_info = agent.train(
                    self.env, num_episodes, print_every=num_episodes // 5
                )

                agent_scores.append(scores)

                if run == 0:
                    training_info["agent"] = agent

            all_results[agent_name] = {
                "scores": agent_scores,
                "mean_scores": np.mean(agent_scores, axis=0),
                "std_scores": np.std(agent_scores, axis=0),
                "training_info": training_info,
            }

        self.results = all_results

        if save_results:
            self.save_comparison_results()

        return all_results

    def save_comparison_results(self, filename="dqn_comparison_results.npy"):
        """Save comparison results to file"""

        serializable_results = {}
        for agent_name, data in self.results.items():
            serializable_results[agent_name] = {
                "scores": data["scores"],
                "mean_scores": data["mean_scores"],
                "std_scores": data["std_scores"],
            }

        np.save(filename, serializable_results)
        print(f"Results saved to {filename}")

    def visualize_comparison(
        self,
        metrics=["learning_curves", "final_performance", "convergence", "stability"],
    ):
        """Create comprehensive visualization of comparison results"""
        if not self.results:
            print("No results to visualize. Run comparison first.")
            return

        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        agent_names = list(self.results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))

        if "learning_curves" in metrics:
            ax = axes[0]
            for i, agent_name in enumerate(agent_names):
                data = self.results[agent_name]
                episodes = range(len(data["mean_scores"]))

                ax.plot(
                    episodes,
                    data["mean_scores"],
                    color=colors[i],
                    label=agent_name,
                    linewidth=2,
                )
                ax.fill_between(
                    episodes,
                    data["mean_scores"] - data["std_scores"],
                    data["mean_scores"] + data["std_scores"],
                    alpha=0.2,
                    color=colors[i],
                )

            ax.set_title("Learning Curves Comparison", fontsize=14, fontweight="bold")
            ax.set_xlabel("Episode", fontsize=12)
            ax.set_ylabel("Episode Reward", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        if "final_performance" in metrics:
            ax = axes[1]
            final_scores = []
            labels = []

            for agent_name in agent_names:
                data = self.results[agent_name]
                final_window = min(50, len(data["mean_scores"]) // 4)
                final_score = data["mean_scores"][-final_window:]
                final_scores.append(final_score)
                labels.append(agent_name)

            ax.boxplot(final_scores, labels=labels)
            ax.set_title(
                "Final Performance Distribution", fontsize=14, fontweight="bold"
            )
            ax.set_ylabel("Episode Reward", fontsize=12)
            ax.grid(True, alpha=0.3)

        if "convergence" in metrics:
            ax = axes[2]
            convergence_episodes = []

            for agent_name in agent_names:
                data = self.results[agent_name]
                scores = data["mean_scores"]

                final_avg = np.mean(scores[-50:])
                threshold = final_avg * 0.9

                conv_episode = next(
                    (i for i, score in enumerate(scores) if score >= threshold),
                    len(scores),
                )
                convergence_episodes.append(conv_episode)

            bars = ax.bar(agent_names, convergence_episodes, color=colors, alpha=0.7)
            ax.set_title("Convergence Speed", fontsize=14, fontweight="bold")
            ax.set_ylabel("Episodes to Convergence", fontsize=12)
            ax.grid(True, alpha=0.3)

            for bar, value in zip(bars, convergence_episodes):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{value}",
                    ha="center",
                    va="bottom",
                )

        if "stability" in metrics:
            ax = axes[3]
            stability_scores = []

            for agent_name in agent_names:
                data = self.results[agent_name]

                final_scores = data["mean_scores"][-50:]
                cv = (
                    np.std(final_scores) / np.mean(final_scores)
                    if np.mean(final_scores) > 0
                    else 0
                )
                stability_scores.append(cv)

            bars = ax.bar(agent_names, stability_scores, color=colors, alpha=0.7)
            ax.set_title("Performance Stability (CV)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Coefficient of Variation", fontsize=12)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def statistical_analysis(self):
        """Perform statistical analysis of results"""
        if not self.results:
            print("No results to analyze. Run comparison first.")
            return

        print("Statistical Analysis of DQN Variants")
        print("=" * 50)

        agent_names = list(self.results.keys())

        final_scores = {}
        for agent_name in agent_names:
            data = self.results[agent_name]
            final_window = min(50, len(data["mean_scores"]) // 4)
            final_score = data["mean_scores"][-final_window:]
            final_scores[agent_name] = final_score

        all_final_scores = [final_scores[name] for name in agent_names]
        f_stat, p_value = stats.f_oneway(*all_final_scores)

        print(f"ANOVA Test Results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("✓ Significant differences found between agents")
        else:
            print("✗ No significant differences found between agents")

        print("\\nPairwise Comparisons:")
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                agent1, agent2 = agent_names[i], agent_names[j]
                t_stat, p_val = stats.ttest_ind(
                    final_scores[agent1], final_scores[agent2]
                )

                print(f"{agent1} vs {agent2}:")
                print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
                if p_val < 0.05:
                    winner = (
                        agent1
                        if np.mean(final_scores[agent1]) > np.mean(final_scores[agent2])
                        else agent2
                    )
                    print(f"  ✓ {winner} significantly better")
                else:
                    print("  - No significant difference")

        print("\\nSummary Statistics:")
        print("-" * 30)
        for agent_name in agent_names:
            scores = final_scores[agent_name]
            print(f"{agent_name}:")
            print(f"  Mean: {np.mean(scores):.2f}")
            print(f"  Std: {np.std(scores):.2f}")
            print(f"  Max: {np.max(scores):.2f}")


class HyperparameterAnalyzer:
    """Analyze hyperparameter sensitivity for DQN variants"""

    def __init__(self, env, state_size, action_size, agent_class):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.agent_class = agent_class

    def analyze_learning_rate(
        self,
        learning_rates=[0.0001, 0.0005, 0.001, 0.005],
        num_episodes=200,
        num_runs=2,
    ):
        """Analyze sensitivity to learning rate"""
        print("Analyzing Learning Rate Sensitivity...")
        print("=" * 40)

        results = {}

        for lr in learning_rates:
            print(f"Testing learning rate: {lr}")
            scores = []

            for run in range(num_runs):
                agent = self.agent_class(self.state_size, self.action_size, lr=lr)
                episode_scores, _ = agent.train(
                    self.env, num_episodes, print_every=num_episodes
                )
                scores.append(episode_scores)

            results[lr] = {
                "scores": scores,
                "mean_final": np.mean([s[-50:] for s in scores]),
                "std_final": np.std([s[-50:] for s in scores]),
            }

        self.visualize_hyperparameter_results(results, "Learning Rate", learning_rates)
        return results

    def analyze_batch_size(
        self, batch_sizes=[16, 32, 64, 128], num_episodes=200, num_runs=2
    ):
        """Analyze sensitivity to batch size"""
        print("Analyzing Batch Size Sensitivity...")
        print("=" * 40)

        results = {}

        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            scores = []

            for run in range(num_runs):
                agent = self.agent_class(
                    self.state_size, self.action_size, batch_size=batch_size
                )
                episode_scores, _ = agent.train(
                    self.env, num_episodes, print_every=num_episodes
                )
                scores.append(episode_scores)

            results[batch_size] = {
                "scores": scores,
                "mean_final": np.mean([s[-50:] for s in scores]),
                "std_final": np.std([s[-50:] for s in scores]),
            }

        self.visualize_hyperparameter_results(results, "Batch Size", batch_sizes)
        return results

    def visualize_hyperparameter_results(self, results, param_name, param_values):
        """Visualize hyperparameter sensitivity results"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        means = [results[val]["mean_final"] for val in param_values]
        stds = [results[val]["std_final"] for val in param_values]

        axes[0].errorbar(
            param_values, means, yerr=stds, fmt="o-", capsize=5, linewidth=2
        )
        axes[0].set_title(f"Final Performance vs {param_name}")
        axes[0].set_xlabel(param_name)
        axes[0].set_ylabel("Final Episode Reward")
        axes[0].grid(True, alpha=0.3)

        for val in param_values:
            scores = results[val]["scores"]
            mean_scores = np.mean(scores, axis=0)
            episodes = range(len(mean_scores))

            axes[1].plot(
                episodes, mean_scores, label=f"{param_name}={val}", linewidth=2
            )

        axes[1].set_title(f"Learning Curves for Different {param_name} Values")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Episode Reward")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        final_scores = [results[val]["scores"][0][-50:] for val in param_values]

        axes[2].boxplot(final_scores, labels=[str(val) for val in param_values])
        axes[2].set_title(f"Performance Distribution vs {param_name}")
        axes[2].set_xlabel(param_name)
        axes[2].set_ylabel("Episode Reward")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class LearningDynamicsAnalyzer:
    """Analyze learning dynamics and algorithmic behavior"""

    def __init__(self):
        self.analysis_results = {}

    def analyze_q_value_dynamics(self, agent, env, num_episodes=10):
        """Analyze how Q-values evolve during learning"""
        print("Analyzing Q-Value Dynamics...")

        q_value_history = []
        td_error_history = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_q_values = []
            episode_td_errors = []

            while not done:

                if hasattr(agent, "get_q_values"):
                    q_values = agent.get_q_values(state)
                else:

                    q_values = np.zeros(self.action_size)

                episode_q_values.append(q_values.copy())

                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if hasattr(agent, "q_network"):
                    with torch.no_grad():
                        state_tensor = (
                            torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                        )
                        next_state_tensor = (
                            torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                        )

                        current_q = agent.q_network(state_tensor)[0, action].item()
                        next_q = agent.target_network(next_state_tensor).max().item()
                        td_error = (
                            reward + agent.gamma * next_q * (1 - done) - current_q
                        )
                        episode_td_errors.append(td_error)

                state = next_state

            q_value_history.append(episode_q_values)
            td_error_history.append(episode_td_errors)

        return {"q_values": q_value_history, "td_errors": td_error_history}

    def visualize_learning_dynamics(self, agent, env):
        """Visualize learning dynamics"""
        dynamics_data = self.analyze_q_value_dynamics(agent, env)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        if dynamics_data["q_values"]:
            first_episode_q = np.array(dynamics_data["q_values"][0])
            time_steps = range(len(first_episode_q))

            for action in range(first_episode_q.shape[1]):
                axes[0, 0].plot(
                    time_steps,
                    first_episode_q[:, action],
                    label=f"Action {action}",
                    linewidth=2,
                )

            axes[0, 0].set_title("Q-Value Evolution (First Episode)")
            axes[0, 0].set_xlabel("Time Step")
            axes[0, 0].set_ylabel("Q-Value")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        if dynamics_data["td_errors"]:
            all_td_errors = [
                td for episode in dynamics_data["td_errors"] for td in episode
            ]

            axes[0, 1].hist(all_td_errors, bins=30, alpha=0.7, color="red")
            axes[0, 1].axvline(x=0, color="black", linestyle="--", alpha=0.5)
            axes[0, 1].set_title("TD Error Distribution")
            axes[0, 1].set_xlabel("TD Error")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].grid(True, alpha=0.3)

        if dynamics_data["q_values"]:
            variances = []
            for episode_q in dynamics_data["q_values"]:
                episode_variances = [np.var(q_vals) for q_vals in episode_q]
                variances.extend(episode_variances)

            axes[1, 0].plot(variances, color="blue", linewidth=2)
            axes[1, 0].set_title("Q-Value Variance Over Time")
            axes[1, 0].set_xlabel("Time Step")
            axes[1, 0].set_ylabel("Q-Value Variance")
            axes[1, 0].grid(True, alpha=0.3)

        if hasattr(agent, "losses") and agent.losses:

            smoothed_losses = uniform_filter1d(agent.losses, size=100)

            axes[1, 1].plot(smoothed_losses, color="green", linewidth=2)
            axes[1, 1].set_title("Training Loss Over Time")
            axes[1, 1].set_xlabel("Training Step")
            axes[1, 1].set_ylabel("Smoothed Loss")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class PerformanceProfiler:
    """Profile computational performance of DQN variants"""

    def __init__(self):
        self.profiling_results = {}

    def profile_agent(self, agent_class, state_size, action_size, **kwargs):
        """Profile computational performance of an agent"""
        import time

        print(f"Profiling {agent_class.__name__}...")

        agent = agent_class(state_size, action_size, **kwargs)

        state = np.random.randn(state_size)
        action = np.random.randint(action_size)
        reward = np.random.randn()
        next_state = np.random.randn(state_size)
        done = False

        for _ in range(100):
            agent.memory.add((state, action, reward, next_state, done))

        start_time = time.time()
        num_steps = 100

        for _ in range(num_steps):
            agent.train_step()

        training_time = time.time() - start_time

        start_time = time.time()
        num_inferences = 1000

        for _ in range(num_inferences):
            agent.act(state)

        inference_time = time.time() - start_time

        results = {
            "agent_type": agent_class.__name__,
            "training_time_per_step": training_time / num_steps,
            "inference_time_per_action": inference_time / num_inferences,
            "memory_usage": len(agent.memory),
            "network_params": sum(p.numel() for p in agent.q_network.parameters()),
        }

        self.profiling_results[agent_class.__name__] = results
        return results

    def compare_performance_profiles(self, agent_configs):
        """Compare performance profiles across agents"""
        print("Comparing Performance Profiles...")
        print("=" * 40)

        results = {}
        for agent_class, kwargs in agent_configs.items():
            results[agent_class.__name__] = self.profile_agent(
                agent_class, 4, 2, **kwargs
            )

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        agent_names = list(results.keys())

        training_times = [
            results[name]["training_time_per_step"] * 1000 for name in agent_names
        ]
        axes[0, 0].bar(agent_names, training_times, alpha=0.7)
        axes[0, 0].set_title("Training Time per Step (ms)")
        axes[0, 0].set_ylabel("Time (ms)")
        axes[0, 0].tick_params(axis="x", rotation=45)

        inference_times = [
            results[name]["inference_time_per_action"] * 1000 for name in agent_names
        ]
        axes[0, 1].bar(agent_names, inference_times, alpha=0.7, color="orange")
        axes[0, 1].set_title("Inference Time per Action (ms)")
        axes[0, 1].set_ylabel("Time (ms)")
        axes[0, 1].tick_params(axis="x", rotation=45)

        param_counts = [results[name]["network_params"] for name in agent_names]
        axes[1, 0].bar(agent_names, param_counts, alpha=0.7, color="green")
        axes[1, 0].set_title("Network Parameters")
        axes[1, 0].set_ylabel("Parameter Count")
        axes[1, 0].tick_params(axis="x", rotation=45)

        efficiency = [
            results[name]["network_params"]
            / (1.0 / results[name]["training_time_per_step"])
            for name in agent_names
        ]
        axes[1, 1].bar(agent_names, efficiency, alpha=0.7, color="red")
        axes[1, 1].set_title("Parameter Efficiency")
        axes[1, 1].set_ylabel("Params / Training Speed")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

        print("\\nPerformance Profile Summary:")
        print("-" * 60)
        print(
            f"{'Agent':<12} {'Train(ms)':<10} {'Infer(ms)':<10} {'Params':<10} {'Memory':<8}"
        )
        print("-" * 60)
        for name in agent_names:
            r = results[name]
            print(
                f"{name:<12} {r['training_time_per_step']*1000:<10.2f} {r['inference_time_per_action']*1000:<10.3f} {r['network_params']:<10} {r['memory_usage']:<8}"
            )


class DQNComparison:
    """Compare different DQN variants"""

    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

    def run_comparison(self, num_episodes=500, num_runs=3):
        """Run comparison between different DQN variants"""
        print("Starting DQN Comparison...")
        print("=" * 60)

        # This is a placeholder - actual implementation would be in the agents
        return [], [], None, None

    def visualize_comparison(self, standard_results, double_results):
        """Visualize comparison results"""
        print("Comparison visualization would go here")


class PerformanceAnalyzer:
    """Analyze performance of DQN agents"""

    def __init__(self):
        pass

    def analyze_performance(self, scores):
        """Analyze performance metrics"""
        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "max": np.max(scores),
            "min": np.min(scores),
        }


if __name__ == "__main__":
    print("DQN Analysis and Visualization Tools")
    print("=" * 45)

    print("Testing analysis components...")

    analyzer = HyperparameterAnalyzer(None, 4, 2, DQNAgent)
    print("✓ Hyperparameter analyzer created")

    dynamics_analyzer = LearningDynamicsAnalyzer()
    print("✓ Learning dynamics analyzer created")

    profiler = PerformanceProfiler()
    print("✓ Performance profiler created")

    print("\\n✓ All analysis tools ready")
    print("✓ Comprehensive DQN evaluation framework available")
