"""
CA7: Comprehensive DQN Experiments and Analysis
===============================================

This script runs comprehensive experiments comparing different DQN variants
and provides detailed analysis of their performance characteristics.

Experiments include:
1. Basic DQN training and evaluation
2. Comparison of DQN variants (Standard, Double, Dueling)
3. Experience replay analysis
4. Target network frequency analysis
5. Overestimation bias analysis
6. Performance benchmarking

Usage:
    python experiments/comprehensive_dqn_analysis.py

Author: CA7 Implementation
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import torch
import warnings
from collections import defaultdict
from typing import Dict, List, Any, Optional
import time

from dqn import (
    DQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent,
    QNetworkVisualization,
    PerformanceAnalyzer,
)

torch.manual_seed(42)
np.random.seed(42)

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
warnings.filterwarnings("ignore")


class ComprehensiveDQNAnalyzer:
    """
    Comprehensive analyzer for DQN variants and techniques
    """

    def __init__(self):
        self.results = {}
        self.analyzers = {
            "visualization": QNetworkVisualization(),
            "performance": PerformanceAnalyzer(),
        }

    def run_basic_dqn_experiment(self):
        """Run basic DQN training experiment"""

        print("=" * 70)
        print("EXPERIMENT 1: Basic DQN Training")
        print("=" * 70)

        env = gym.make("CartPole-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        print(f"Environment: CartPole-v1")
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"Goal: Balance pole for as long as possible (max 500 steps)")
        print()

        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=1e-3,  # Learning rate
            gamma=0.99,  # Discount factor
            epsilon_start=1.0,  # Initial exploration
            epsilon_end=0.01,  # Final exploration
            epsilon_decay=0.995,  # Exploration decay
            buffer_size=10000,  # Experience replay buffer size
            batch_size=64,  # Training batch size
            target_update_freq=100,  # Target network update frequency
        )

        num_episodes = 200
        max_steps_per_episode = 500

        print("Training Configuration:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Max steps per episode: {max_steps_per_episode}")
        print(f"  Learning rate: {agent.optimizer.param_groups[0]['lr']}")
        print(f"  Gamma: {agent.gamma}")
        print(f"  Epsilon decay: {agent.epsilon_decay}")
        print()

        print("Starting training...")
        print("-" * 50)

        episode_rewards = []
        training_start_time = time.time()

        for episode in range(num_episodes):
            reward, steps = agent.train_episode(env, max_steps=max_steps_per_episode)
            episode_rewards.append(reward)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                elapsed_time = time.time() - training_start_time
                print(
                    f"Episode {episode+1:3d} | Avg Reward: {avg_reward:6.1f} | "
                    f"Epsilon: {agent.epsilon:.3f} | Time: {elapsed_time:.1f}s"
                )

        training_time = time.time() - training_start_time
        print("-" * 50)
        print(f"Training completed in {training_time:.1f} seconds!")
        print()

        print("Final Evaluation:")
        eval_results = agent.evaluate(env, num_episodes=20)
        print(
            f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}"
        )
        print(f"Success Rate: {(eval_results['mean_reward'] >= 195):.1%}")

        env.close()

        self.results["basic_dqn"] = {
            "agent": agent,
            "rewards": episode_rewards,
            "eval_performance": eval_results,
            "training_time": training_time,
        }

        return agent, episode_rewards

    def compare_dqn_variants(self):
        """Compare all DQN variants"""

        print("=" * 70)
        print("EXPERIMENT 2: DQN Variants Comparison")
        print("=" * 70)

        env = gym.make("CartPole-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        variants = {
            "Standard DQN": DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=1e-3,
                epsilon_decay=0.995,
                buffer_size=15000,
            ),
            "Double DQN": DoubleDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=1e-3,
                epsilon_decay=0.995,
                buffer_size=15000,
            ),
            "Dueling DQN (Mean)": DuelingDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                dueling_type="mean",
                lr=1e-3,
                epsilon_decay=0.995,
                buffer_size=15000,
            ),
            "Dueling DQN (Max)": DuelingDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                dueling_type="max",
                lr=1e-3,
                epsilon_decay=0.995,
                buffer_size=15000,
            ),
        }

        results = {}
        num_episodes = 150

        for name, agent in variants.items():
            print(f"\nTraining {name}...")
            episode_rewards = []

            start_time = time.time()
            for episode in range(num_episodes):
                reward, _ = agent.train_episode(env, max_steps=500)
                episode_rewards.append(reward)

                if (episode + 1) % 50 == 0:
                    avg_reward = np.mean(episode_rewards[-50:])
                    elapsed = time.time() - start_time
                    print(
                        f"  Episode {episode+1}: Avg Reward = {avg_reward:.1f} "
                        f"({elapsed:.1f}s)"
                    )

            eval_results = agent.evaluate(env, num_episodes=15)
            training_time = time.time() - start_time

            results[name] = {
                "agent": agent,
                "rewards": episode_rewards,
                "eval_performance": eval_results,
                "final_performance": np.mean(episode_rewards[-20:]),
                "training_time": training_time,
            }

        self._plot_variant_comparison(results)

        env.close()
        self.results["variant_comparison"] = results
        return results

    def _plot_variant_comparison(self, results):
        """Plot comprehensive comparison of DQN variants"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        colors = ["blue", "red", "green", "orange"]

        ax = axes[0, 0]
        for i, (variant, data) in enumerate(results.items()):
            rewards = data["rewards"]
            smoothed = pd.Series(rewards).rolling(10).mean()
            ax.plot(smoothed, label=variant, color=colors[i], linewidth=2)
            ax.fill_between(range(len(smoothed)), smoothed, alpha=0.3, color=colors[i])

        ax.set_title("Learning Curves Comparison")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (Smoothed)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        variant_names = list(results.keys())
        final_perfs = [results[v]["final_performance"] for v in variant_names]
        eval_means = [
            results[v]["eval_performance"]["mean_reward"] for v in variant_names
        ]
        eval_stds = [
            results[v]["eval_performance"]["std_reward"] for v in variant_names
        ]

        x = np.arange(len(variant_names))
        width = 0.35

        ax.bar(
            x - width / 2,
            final_perfs,
            width,
            label="Training Performance",
            alpha=0.7,
            color=colors,
        )
        ax.bar(
            x + width / 2,
            eval_means,
            width,
            yerr=eval_stds,
            label="Evaluation Performance",
            alpha=0.7,
            color=colors,
        )

        ax.set_title("Performance Comparison")
        ax.set_ylabel("Average Reward")
        ax.set_xticks(x)
        ax.set_xticklabels(variant_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        training_times = [results[v]["training_time"] for v in variant_names]

        bars = ax.bar(variant_names, training_times, alpha=0.7, color=colors)
        ax.set_title("Training Time Comparison")
        ax.set_ylabel("Time (seconds)")
        ax.set_xticklabels(variant_names, rotation=45, ha="right")

        for bar, time_val in zip(bars, training_times):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{time_val:.1f}s",
                ha="center",
                va="bottom",
            )

        ax = axes[1, 0]
        for i, (variant, data) in enumerate(results.items()):
            agent = data["agent"]
            if hasattr(agent, "q_values_history") and agent.q_values_history:
                smoothed_q = pd.Series(agent.q_values_history).rolling(50).mean()
                ax.plot(
                    smoothed_q,
                    label=f"{variant} Q-values",
                    color=colors[i],
                    linewidth=2,
                )

        ax.set_title("Q-Value Evolution")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Average Q-Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for i, (variant, data) in enumerate(results.items()):
            agent = data["agent"]
            if hasattr(agent, "losses") and agent.losses:
                losses = agent.losses
                if len(losses) > 50:
                    smoothed_loss = pd.Series(losses).rolling(50).mean()
                    ax.plot(
                        smoothed_loss,
                        label=f"{variant} Loss",
                        color=colors[i],
                        linewidth=2,
                        alpha=0.7,
                    )

        ax.set_title("Training Loss Comparison")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MSE Loss (Smoothed)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]

        sample_state = [0.1, 0.1, 0.1, 0.1]  # Example CartPole state

        q_values_comparison = {}
        for variant, data in results.items():
            agent = data["agent"]
            q_vals = agent.get_q_values(sample_state)
            q_values_comparison[variant] = q_vals

        if q_values_comparison:
            x = np.arange(len(q_vals))
            width = 0.2

            for i, (variant, q_vals) in enumerate(q_values_comparison.items()):
                ax.bar(
                    x + i * width,
                    q_vals,
                    width,
                    label=variant,
                    alpha=0.7,
                    color=colors[i],
                )

            ax.set_title("Q-Values for Sample State")
            ax.set_xlabel("Actions")
            ax.set_ylabel("Q-Value")
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels([f"Action {i}" for i in range(len(q_vals))])
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("dqn_variant_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

    def analyze_experience_replay(self):
        """Analyze the impact of experience replay strategies"""

        print("=" * 70)
        print("EXPERIMENT 3: Experience Replay Analysis")
        print("=" * 70)

        env = gym.make("CartPole-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        strategies = {
            "No Replay": {"buffer_size": 1, "batch_size": 1},
            "Small Buffer (1K)": {"buffer_size": 1000, "batch_size": 32},
            "Medium Buffer (5K)": {"buffer_size": 5000, "batch_size": 64},
            "Large Buffer (20K)": {"buffer_size": 20000, "batch_size": 64},
        }

        results = {}
        num_episodes = 100

        for strategy_name, config in strategies.items():
            print(f"\nTesting {strategy_name}...")

            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                buffer_size=config["buffer_size"],
                batch_size=config["batch_size"],
                lr=1e-3,
                epsilon_decay=0.99,
                target_update_freq=100,
            )

            episode_rewards = []
            losses = []

            for episode in range(num_episodes):
                reward, _ = agent.train_episode(env, max_steps=500)
                episode_rewards.append(reward)

                if len(agent.losses) > len(losses):
                    losses.extend(agent.losses[len(losses) :])

                if (episode + 1) % 25 == 0:
                    avg_reward = np.mean(episode_rewards[-25:])
                    print(f"  Episode {episode+1}: Avg Reward = {avg_reward:.1f}")

            results[strategy_name] = {
                "rewards": episode_rewards,
                "losses": losses,
                "final_performance": np.mean(episode_rewards[-20:]),
            }

        self._plot_replay_analysis(results)

        env.close()
        self.results["replay_analysis"] = results
        return results

    def _plot_replay_analysis(self, results):
        """Plot experience replay analysis"""

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        colors = ["red", "blue", "green", "orange"]

        ax = axes[0]
        for i, (strategy, data) in enumerate(results.items()):
            rewards = data["rewards"]
            smoothed = pd.Series(rewards).rolling(10).mean()
            ax.plot(smoothed, label=strategy, color=colors[i], linewidth=2)

        ax.set_title("Learning Curves by Replay Strategy")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (Smoothed)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        strategies_list = list(results.keys())
        final_perfs = [results[s]["final_performance"] for s in strategies_list]

        bars = ax.bar(strategies_list, final_perfs, alpha=0.7, color=colors)
        ax.set_title("Final Performance Comparison")
        ax.set_ylabel("Average Reward (Last 20 Episodes)")
        ax.set_xticklabels(strategies_list, rotation=45, ha="right")

        for bar, perf in zip(bars, final_perfs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{perf:.1f}",
                ha="center",
                va="bottom",
            )

        ax = axes[2]
        for i, (strategy, data) in enumerate(results.items()):
            if len(data["losses"]) > 10:
                losses = data["losses"]
                smoothed_losses = pd.Series(losses).rolling(50).mean()
                ax.plot(
                    smoothed_losses,
                    label=strategy,
                    color=colors[i],
                    linewidth=2,
                    alpha=0.7,
                )

        ax.set_title("Training Loss Comparison")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MSE Loss (Smoothed)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("experience_replay_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def run_theoretical_analysis(self):
        """Run theoretical concept demonstrations"""

        print("=" * 70)
        print("EXPERIMENT 4: Theoretical Concept Analysis")
        print("=" * 70)

        visualizer = self.analyzers["visualization"]

        print("1. Visualizing Core Q-Learning Concepts...")
        visualizer.visualize_q_learning_concepts()

        print("\n2. Demonstrating Overestimation Bias...")
        visualizer.demonstrate_overestimation_bias()

        print("\nTheoretical analysis completed!")

    def generate_comprehensive_report(self):
        """Generate comprehensive experiment report"""

        print("=" * 70)
        print("COMPREHENSIVE EXPERIMENT REPORT")
        print("=" * 70)

        if "basic_dqn" in self.results:
            basic = self.results["basic_dqn"]
            print(f"\nBasic DQN Results:")
            print(
                f"  Final Training Performance: {basic['eval_performance']['mean_reward']:.2f} ± {basic['eval_performance']['std_reward']:.2f}"
            )
            print(f"  Training Time: {basic['training_time']:.1f} seconds")
            print(
                f"  Success Rate: {(basic['eval_performance']['mean_reward'] >= 195):.1%}"
            )

        if "variant_comparison" in self.results:
            variants = self.results["variant_comparison"]
            print(f"\nDQN Variants Comparison:")
            for name, data in variants.items():
                perf = data["eval_performance"]
                time_taken = data["training_time"]
                print(
                    f"  {name}: {perf['mean_reward']:.1f} ± {perf['std_reward']:.1f} "
                    f"({time_taken:.1f}s)"
                )

        if "replay_analysis" in self.results:
            replay = self.results["replay_analysis"]
            print(f"\nExperience Replay Analysis:")
            for name, data in replay.items():
                perf = data["final_performance"]
                print(f"  {name}: {perf:.1f} final reward")

        print(f"\n{'='*70}")
        print("Key Findings:")
        print("• Experience replay is crucial for stable DQN training")
        print("• Double DQN reduces overestimation bias")
        print("• Dueling DQN improves value estimation efficiency")
        print("• Larger replay buffers generally improve performance")
        print("• All variants can solve CartPole-v1 with proper tuning")
        print(f"{'='*70}")

    def run_all_experiments(self):
        """Run all experiments in sequence"""

        print("Starting Comprehensive DQN Analysis Suite")
        print("This will take several minutes to complete...")
        print()

        self.run_basic_dqn_experiment()
        self.compare_dqn_variants()
        self.analyze_experience_replay()
        self.run_theoretical_analysis()

        self.generate_comprehensive_report()

        print("\nAll experiments completed!")
        print("Results saved as PNG files in the current directory.")


def main():
    """Main function to run comprehensive DQN analysis"""

    analyzer = ComprehensiveDQNAnalyzer()

    analyzer.run_all_experiments()


if __name__ == "__main__":
    main()
