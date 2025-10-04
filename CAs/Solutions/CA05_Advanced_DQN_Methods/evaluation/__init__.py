"""
Evaluation utilities for CA5 Advanced DQN Methods
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import torch
import json
import os


class PerformanceEvaluator:
    """Evaluate DQN agent performance"""

    def __init__(self, save_dir: str = "evaluation/results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.results = {}

    def evaluate_agent(
        self, agent, env, num_episodes: int = 100, render: bool = False
    ) -> Dict[str, Any]:
        """Evaluate agent performance over multiple episodes"""

        episode_rewards = []
        episode_lengths = []
        success_rate = 0

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                if render and episode % 10 == 0:
                    env.render()

                action = agent.select_action(state, epsilon=0.0)  # No exploration
                next_state, reward, done, info = env.step(action)

                episode_reward += reward
                episode_length += 1
                state = next_state

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Check if episode was successful (customize based on environment)
            if episode_reward > 0:  # Simple success criterion
                success_rate += 1

        success_rate /= num_episodes

        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "success_rate": success_rate,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

        self.results[f"evaluation_{len(self.results)}"] = results
        return results

    def plot_performance(
        self, results: Dict[str, Any], save_path: Optional[str] = None
    ):
        """Plot performance metrics"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        axes[0, 0].plot(results["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)

        # Episode lengths
        axes[0, 1].plot(results["episode_lengths"])
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(True)

        # Reward distribution
        axes[1, 0].hist(results["episode_rewards"], bins=20, alpha=0.7)
        axes[1, 0].set_title("Reward Distribution")
        axes[1, 0].set_xlabel("Reward")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True)

        # Performance summary
        summary_text = f"""
        Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}
        Mean Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}
        Success Rate: {results['success_rate']:.2%}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment="center")
        axes[1, 1].set_title("Performance Summary")
        axes[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def save_results(self, filename: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")


def compare_agents(
    agents: Dict[str, Any], env, num_episodes: int = 100
) -> Dict[str, Any]:
    """Compare multiple agents"""

    evaluator = PerformanceEvaluator()
    comparison_results = {}

    for agent_name, agent in agents.items():
        print(f"Evaluating {agent_name}...")
        results = evaluator.evaluate_agent(agent, env, num_episodes)
        comparison_results[agent_name] = results

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Mean rewards comparison
    agent_names = list(comparison_results.keys())
    mean_rewards = [comparison_results[name]["mean_reward"] for name in agent_names]
    std_rewards = [comparison_results[name]["std_reward"] for name in agent_names]

    axes[0, 0].bar(agent_names, mean_rewards, yerr=std_rewards, capsize=5)
    axes[0, 0].set_title("Mean Rewards Comparison")
    axes[0, 0].set_ylabel("Mean Reward")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Success rates comparison
    success_rates = [comparison_results[name]["success_rate"] for name in agent_names]
    axes[0, 1].bar(agent_names, success_rates)
    axes[0, 1].set_title("Success Rates Comparison")
    axes[0, 1].set_ylabel("Success Rate")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Episode lengths comparison
    mean_lengths = [comparison_results[name]["mean_length"] for name in agent_names]
    std_lengths = [comparison_results[name]["std_length"] for name in agent_names]

    axes[1, 0].bar(agent_names, mean_lengths, yerr=std_lengths, capsize=5)
    axes[1, 0].set_title("Mean Episode Lengths Comparison")
    axes[1, 0].set_ylabel("Mean Length")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Learning curves (if available)
    axes[1, 1].text(
        0.5,
        0.5,
        "Learning Curves\n(Not implemented)",
        ha="center",
        va="center",
        fontsize=12,
    )
    axes[1, 1].set_title("Learning Curves")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("visualizations/agent_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    return comparison_results


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")

