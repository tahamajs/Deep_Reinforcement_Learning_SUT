import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from itertools import product
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

from .utils import device
from ..agents.reinforce import REINFORCEAgent
from ..agents.actor_critic import ActorCriticAgent, A2CAgent
from ..agents.ppo import PPOAgent
from ..agents.baseline_reinforce import BaselineREINFORCEAgent


class HyperparameterTuner:
    """Hyperparameter tuning for policy gradient methods"""

    def __init__(self, env_name: str = "CartPole-v1"):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

    def tune_learning_rates(
        self,
        lr_range: List[float] = [1e-5, 1e-4, 1e-3, 1e-2],
        num_episodes: int = 100,
        num_seeds: int = 3,
    ) -> Dict[str, Any]:
        """Tune learning rates for different algorithms"""

        print("=" * 60)
        print("Learning Rate Tuning")
        print("=" * 60)

        algorithms = {
            "REINFORCE": REINFORCEAgent,
            "Actor-Critic": ActorCriticAgent,
            "PPO": PPOAgent,
        }

        results = {}

        for alg_name, agent_class in algorithms.items():
            print(f"\nTuning {alg_name}...")
            alg_results = []

            for lr in lr_range:
                print(f"  Testing LR = {lr:.0e}")
                seed_results = []

                for seed in range(num_seeds):
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    if alg_name == "REINFORCE":
                        agent = agent_class(self.state_dim, self.action_dim, lr=lr)
                    elif alg_name == "Actor-Critic":
                        agent = agent_class(
                            self.state_dim, self.action_dim, lr_actor=lr, lr_critic=lr
                        )
                    else:  # PPO
                        agent = agent_class(self.state_dim, self.action_dim, lr=lr)

                    # Train agent
                    episode_rewards = []
                    for episode in range(num_episodes):
                        reward, _ = agent.train_episode(self.env)
                        episode_rewards.append(reward)

                    # Calculate final performance
                    final_performance = np.mean(episode_rewards[-20:])
                    seed_results.append(final_performance)

                avg_performance = np.mean(seed_results)
                std_performance = np.std(seed_results)

                alg_results.append(
                    {
                        "lr": lr,
                        "performance": avg_performance,
                        "std": std_performance,
                        "seed_results": seed_results,
                    }
                )

            results[alg_name] = alg_results

        self.env.close()
        self._plot_lr_tuning_results(results, lr_range)

        return results

    def tune_ppo_parameters(
        self,
        clip_ratios: List[float] = [0.1, 0.2, 0.3],
        k_epochs: List[int] = [3, 5, 10],
        num_episodes: int = 150,
        num_seeds: int = 2,
    ) -> Dict[str, Any]:
        """Tune PPO-specific hyperparameters"""

        print("=" * 60)
        print("PPO Hyperparameter Tuning")
        print("=" * 60)

        results = []

        for clip_ratio, k_epoch in product(clip_ratios, k_epochs):
            print(f"Testing clip_ratio={clip_ratio}, k_epochs={k_epoch}")
            seed_results = []

            for seed in range(num_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                agent = PPOAgent(
                    self.state_dim,
                    self.action_dim,
                    eps_clip=clip_ratio,
                    k_epochs=k_epoch,
                )

                # Train agent
                episode_rewards = []
                for episode in range(num_episodes):
                    reward, _ = agent.train_episode(self.env)
                    episode_rewards.append(reward)

                final_performance = np.mean(episode_rewards[-20:])
                seed_results.append(final_performance)

            avg_performance = np.mean(seed_results)
            std_performance = np.std(seed_results)

            results.append(
                {
                    "clip_ratio": clip_ratio,
                    "k_epochs": k_epoch,
                    "performance": avg_performance,
                    "std": std_performance,
                    "seed_results": seed_results,
                }
            )

        self.env.close()
        self._plot_ppo_tuning_results(results)

        return results

    def _plot_lr_tuning_results(self, results: Dict[str, Any], lr_range: List[float]):
        """Plot learning rate tuning results"""

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Performance vs Learning Rate
        ax = axes[0]
        colors = ["blue", "red", "green"]

        for i, (alg_name, alg_results) in enumerate(results.items()):
            lrs = [r["lr"] for r in alg_results]
            performances = [r["performance"] for r in alg_results]
            stds = [r["std"] for r in alg_results]

            ax.errorbar(
                lrs,
                performances,
                yerr=stds,
                label=alg_name,
                color=colors[i],
                marker="o",
                linewidth=2,
                capsize=5,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Final Performance")
        ax.set_title("Learning Rate Tuning Results")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Best learning rates
        ax = axes[1]
        best_lrs = []
        best_performances = []
        alg_names = []

        for alg_name, alg_results in results.items():
            best_result = max(alg_results, key=lambda x: x["performance"])
            best_lrs.append(best_result["lr"])
            best_performances.append(best_result["performance"])
            alg_names.append(alg_name)

        bars = ax.bar(
            alg_names, best_performances, color=colors[: len(alg_names)], alpha=0.7
        )
        ax.set_ylabel("Best Performance")
        ax.set_title("Best Learning Rate Performance")
        ax.grid(True, alpha=0.3)

        # Add best LR values on bars
        for bar, lr in zip(bars, best_lrs):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"LR={lr:.0e}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()

        # Print summary
        print("\n" + "=" * 50)
        print("LEARNING RATE TUNING SUMMARY")
        print("=" * 50)

        for alg_name, alg_results in results.items():
            best_result = max(alg_results, key=lambda x: x["performance"])
            print(f"\n{alg_name}:")
            print(f"  Best LR: {best_result['lr']:.0e}")
            print(
                f"  Best Performance: {best_result['performance']:.2f} ± {best_result['std']:.2f}"
            )

    def _plot_ppo_tuning_results(self, results: List[Dict[str, Any]]):
        """Plot PPO hyperparameter tuning results"""

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Performance heatmap
        ax = axes[0]

        clip_ratios = sorted(list(set([r["clip_ratio"] for r in results])))
        k_epochs = sorted(list(set([r["k_epochs"] for r in results])))

        performance_matrix = np.zeros((len(clip_ratios), len(k_epochs)))

        for result in results:
            i = clip_ratios.index(result["clip_ratio"])
            j = k_epochs.index(result["k_epochs"])
            performance_matrix[i, j] = result["performance"]

        im = ax.imshow(performance_matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(k_epochs)))
        ax.set_xticklabels(k_epochs)
        ax.set_yticks(range(len(clip_ratios)))
        ax.set_yticklabels(clip_ratios)
        ax.set_xlabel("K Epochs")
        ax.set_ylabel("Clip Ratio")
        ax.set_title("PPO Hyperparameter Performance")

        # Add text annotations
        for i in range(len(clip_ratios)):
            for j in range(len(k_epochs)):
                text = ax.text(
                    j,
                    i,
                    f"{performance_matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                )

        plt.colorbar(im, ax=ax, label="Performance")

        # Best parameters
        ax = axes[1]
        best_result = max(results, key=lambda x: x["performance"])

        ax.bar(
            ["Clip Ratio", "K Epochs"],
            [best_result["clip_ratio"], best_result["k_epochs"]],
            color=["blue", "red"],
            alpha=0.7,
        )
        ax.set_ylabel("Value")
        ax.set_title("Best PPO Parameters")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary
        print("\n" + "=" * 50)
        print("PPO HYPERPARAMETER TUNING SUMMARY")
        print("=" * 50)
        print(f"Best Clip Ratio: {best_result['clip_ratio']}")
        print(f"Best K Epochs: {best_result['k_epochs']}")
        print(
            f"Best Performance: {best_result['performance']:.2f} ± {best_result['std']:.2f}"
        )


class PolicyGradientBenchmark:
    """Comprehensive benchmarking for policy gradient methods"""

    def __init__(self):
        self.environments = ["CartPole-v1", "LunarLander-v2"]
        self.algorithms = {
            "REINFORCE": REINFORCEAgent,
            "REINFORCE+Baseline": BaselineREINFORCEAgent,
            "Actor-Critic": ActorCriticAgent,
            "A2C": A2CAgent,
            "PPO": PPOAgent,
        }

    def run_benchmark(
        self, num_episodes: int = 200, num_seeds: int = 3
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across environments and algorithms"""

        print("=" * 70)
        print("Policy Gradient Methods Benchmark")
        print("=" * 70)

        results = {}

        for env_name in self.environments:
            print(f"\nBenchmarking on {env_name}...")
            env = gym.make(env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            env_results = {}

            for alg_name, agent_class in self.algorithms.items():
                print(f"  Testing {alg_name}...")
                seed_results = []

                for seed in range(num_seeds):
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    # Create agent with default parameters
                    if alg_name == "REINFORCE":
                        agent = agent_class(state_dim, action_dim)
                    elif alg_name == "REINFORCE+Baseline":
                        agent = agent_class(
                            state_dim, action_dim, baseline_type="value_function"
                        )
                    elif alg_name == "Actor-Critic":
                        agent = agent_class(state_dim, action_dim)
                    elif alg_name == "A2C":
                        agent = agent_class(state_dim, action_dim, n_steps=5)
                    else:  # PPO
                        agent = agent_class(state_dim, action_dim)

                    # Train agent
                    episode_rewards = []
                    for episode in range(num_episodes):
                        reward, _ = agent.train_episode(env)
                        episode_rewards.append(reward)

                    # Calculate metrics
                    final_performance = np.mean(episode_rewards[-20:])
                    convergence_episode = self._find_convergence_episode(
                        episode_rewards
                    )
                    stability = (
                        np.std(episode_rewards[-50:])
                        if len(episode_rewards) >= 50
                        else 0
                    )

                    seed_results.append(
                        {
                            "final_performance": final_performance,
                            "convergence_episode": convergence_episode,
                            "stability": stability,
                            "episode_rewards": episode_rewards,
                        }
                    )

                # Aggregate results
                env_results[alg_name] = {
                    "mean_final_performance": np.mean(
                        [r["final_performance"] for r in seed_results]
                    ),
                    "std_final_performance": np.std(
                        [r["final_performance"] for r in seed_results]
                    ),
                    "mean_convergence_episode": np.mean(
                        [r["convergence_episode"] for r in seed_results]
                    ),
                    "mean_stability": np.mean([r["stability"] for r in seed_results]),
                    "seed_results": seed_results,
                }

            results[env_name] = env_results
            env.close()

        self._plot_benchmark_results(results)

        return results

    def _find_convergence_episode(
        self, episode_rewards: List[float], threshold: float = 0.8
    ) -> int:
        """Find episode where performance reaches threshold of final performance"""
        if len(episode_rewards) < 10:
            return len(episode_rewards)

        final_performance = np.mean(episode_rewards[-20:])
        target_performance = threshold * final_performance

        for i in range(10, len(episode_rewards)):
            if np.mean(episode_rewards[i - 5 : i]) >= target_performance:
                return i

        return len(episode_rewards)

    def _plot_benchmark_results(self, results: Dict[str, Any]):
        """Plot comprehensive benchmark results"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Performance comparison
        ax = axes[0, 0]
        env_names = list(results.keys())
        alg_names = list(results[env_names[0]].keys())

        x = np.arange(len(alg_names))
        width = 0.35

        for i, env_name in enumerate(env_names):
            env_results = results[env_name]
            performances = [
                env_results[alg]["mean_final_performance"] for alg in alg_names
            ]
            stds = [env_results[alg]["std_final_performance"] for alg in alg_names]

            ax.bar(
                x + i * width,
                performances,
                width,
                label=env_name,
                yerr=stds,
                capsize=5,
                alpha=0.7,
            )

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Final Performance")
        ax.set_title("Performance Comparison")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(alg_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sample efficiency
        ax = axes[0, 1]
        for i, env_name in enumerate(env_names):
            env_results = results[env_name]
            convergence_episodes = [
                env_results[alg]["mean_convergence_episode"] for alg in alg_names
            ]

            ax.plot(
                alg_names,
                convergence_episodes,
                "o-",
                label=env_name,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Episodes to Convergence")
        ax.set_title("Sample Efficiency")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Stability
        ax = axes[0, 2]
        for i, env_name in enumerate(env_names):
            env_results = results[env_name]
            stabilities = [env_results[alg]["mean_stability"] for alg in alg_names]

            ax.bar(
                [f"{alg}\n({env_name})" for alg in alg_names],
                stabilities,
                alpha=0.7,
                color=["blue", "red"][i],
            )

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Training Stability (Lower = Better)")
        ax.set_title("Training Stability")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        # Learning curves for best environment
        ax = axes[1, 0]
        best_env = env_names[0]  # Use first environment
        env_results = results[best_env]

        for alg_name in alg_names:
            seed_results = env_results[alg_name]["seed_results"]
            all_rewards = [r["episode_rewards"] for r in seed_results]
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)

            episodes = range(len(mean_rewards))
            ax.plot(episodes, mean_rewards, label=alg_name, linewidth=2)
            ax.fill_between(
                episodes,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.3,
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.set_title(f"Learning Curves - {best_env}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Performance vs Stability trade-off
        ax = axes[1, 1]
        for env_name in env_names:
            env_results = results[env_name]
            performances = [
                env_results[alg]["mean_final_performance"] for alg in alg_names
            ]
            stabilities = [env_results[alg]["mean_stability"] for alg in alg_names]

            ax.scatter(stabilities, performances, label=env_name, s=100, alpha=0.7)

            for i, alg in enumerate(alg_names):
                ax.annotate(
                    alg,
                    (stabilities[i], performances[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax.set_xlabel("Training Stability (Lower = Better)")
        ax.set_ylabel("Final Performance")
        ax.set_title("Performance vs Stability Trade-off")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Summary table
        ax = axes[1, 2]
        ax.axis("off")

        # Create summary table
        summary_data = []
        for env_name in env_names:
            env_results = results[env_name]
            for alg_name in alg_names:
                summary_data.append(
                    [
                        env_name,
                        alg_name,
                        f"{env_results[alg_name]['mean_final_performance']:.1f}",
                        f"{env_results[alg_name]['mean_convergence_episode']:.0f}",
                        f"{env_results[alg_name]['mean_stability']:.1f}",
                    ]
                )

        table = ax.table(
            cellText=summary_data,
            colLabels=[
                "Environment",
                "Algorithm",
                "Performance",
                "Convergence",
                "Stability",
            ],
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        ax.set_title("Benchmark Summary", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.show()

        # Print summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        for env_name in env_names:
            print(f"\n{env_name}:")
            env_results = results[env_name]

            # Find best algorithm for each metric
            best_performance = max(
                alg_names, key=lambda x: env_results[x]["mean_final_performance"]
            )
            best_efficiency = min(
                alg_names, key=lambda x: env_results[x]["mean_convergence_episode"]
            )
            best_stability = min(
                alg_names, key=lambda x: env_results[x]["mean_stability"]
            )

            print(
                f"  Best Performance: {best_performance} ({env_results[best_performance]['mean_final_performance']:.1f})"
            )
            print(
                f"  Best Efficiency: {best_efficiency} ({env_results[best_efficiency]['mean_convergence_episode']:.0f} episodes)"
            )
            print(
                f"  Best Stability: {best_stability} ({env_results[best_stability]['mean_stability']:.1f})"
            )


# Example usage functions
def run_hyperparameter_tuning_example():
    """Example of running hyperparameter tuning"""
    tuner = HyperparameterTuner("CartPole-v1")

    print("Running learning rate tuning...")
    lr_results = tuner.tune_learning_rates()

    print("\nRunning PPO parameter tuning...")
    ppo_results = tuner.tune_ppo_parameters()

    return lr_results, ppo_results


def run_benchmark_example():
    """Example of running comprehensive benchmark"""
    benchmark = PolicyGradientBenchmark()

    print("Running comprehensive benchmark...")
    results = benchmark.run_benchmark(num_episodes=150, num_seeds=2)

    return results


if __name__ == "__main__":
    # Run examples
    lr_results, ppo_results = run_hyperparameter_tuning_example()
    benchmark_results = run_benchmark_example()
