"""
evaluation.py - Functions for evaluating and comparing reinforcement learning agents.

This module provides utilities to evaluate the performance of TD learning agents,
compare multiple agents, and analyze their learning characteristics. It helps
in quantifying agent performance, stability, and convergence.
"""

from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Assuming agents, environments, and config are available in the src package
from .environments import GridWorld
from .agents import BaseAgent, TD0Agent, QLearningAgent, SARSAAgent, RandomPolicy
from .config import ExperimentConfig, VisualizationConfig

def evaluate_agent(
    agent: BaseAgent,
    env: GridWorld,
    num_episodes: int = ExperimentConfig.EVAL_EPISODES,
    save_results: bool = False,
    save_dir: str = VisualizationConfig.SAVE_DIR,
    filename: str = "evaluation_results.json",
) -> Dict[str, Any]:
    """
    Evaluates a given agent's policy in the environment.

    Args:
        agent (BaseAgent): The agent to evaluate.
        env (GridWorld): The environment instance.
        num_episodes (int): Number of episodes for evaluation.
        save_results (bool): Whether to save the evaluation results to a JSON file.
        save_dir (str): Directory to save results if `save_results` is True.
        filename (str): Filename for the saved JSON results.

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics.
    """
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    evaluation_metrics = agent.evaluate_policy(num_episodes=num_episodes)
    
    if save_results:
        # Ensure save_dir exists
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save using model_utils.export_results (assuming it exists or define here)
        # For now, let's just print a message. Proper saving handled by visualization/utils module.
        print(f"Evaluation results can be saved, but will be handled by utility functions.")

    print(f"  • Average Reward: {evaluation_metrics['avg_reward']:.2f} ± {evaluation_metrics['std_reward']:.2f}")
    print(f"  • Success Rate: {evaluation_metrics['success_rate']*100:.1f}%")
    print(f"  • Average Steps: {evaluation_metrics['avg_steps']:.1f}")
    return evaluation_metrics

def compare_agents(
    agents: Dict[str, BaseAgent],
    env: GridWorld,
    num_episodes: int = ExperimentConfig.EVAL_EPISODES,
    save_dir: str = VisualizationConfig.SAVE_DIR,
    filepath: str = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compares the performance of multiple agents.

    Args:
        agents (Dict[str, BaseAgent]): A dictionary of agent names to agent instances.
        env (GridWorld): The environment instance.
        num_episodes (int): Number of episodes for evaluation per agent.
        save_dir (str): Directory to save plots.
        filepath (str, optional): Path to save the comparison plot.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary where keys are agent names and values are their evaluation metrics.
    """
    print(f"\n--- Comparing {len(agents)} Agents ({num_episodes} evaluation episodes each) ---")
    comparison_results = {}
    all_rewards = []
    all_labels = []

    for name, agent in agents.items():
        print(f"  Evaluating {name}...")
        metrics = agent.evaluate_policy(num_episodes=num_episodes)
        comparison_results[name] = metrics
        all_rewards.extend(metrics["rewards_per_episode"])
        all_labels.extend([name] * num_episodes)
        print(f"    Avg Reward: {metrics['avg_reward']:.2f}, Success Rate: {metrics['success_rate']*100:.1f}%")

    # Plotting comparison
    plt.figure(figsize=VisualizationConfig.FIGURE_SIZE)
    df = pd.DataFrame({"Reward": all_rewards, "Agent": all_labels})
    sns.boxplot(x="Agent", y="Reward", data=df)
    plt.title("Agent Performance Comparison", fontsize=VisualizationConfig.FONT_SIZE + 2)
    plt.xlabel("Agent", fontsize=VisualizationConfig.FONT_SIZE)
    plt.ylabel("Total Reward Per Episode", fontsize=VisualizationConfig.FONT_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()
    
    print("Agent comparison completed.")
    return comparison_results

def analyze_performance(
    agent: BaseAgent,
    save_dir: str = VisualizationConfig.SAVE_DIR,
    filepath_prefix: str = "performance_analysis",
) -> Dict[str, Any]:
    """
    Analyzes and visualizes the learning performance of a single agent.

    Args:
        agent (BaseAgent): The trained agent to analyze.
        save_dir (str): Directory to save plots.
        filepath_prefix (str): Prefix for saved plot filenames.

    Returns:
        Dict[str, Any]: Summary of performance characteristics.
    """
    print(f"\n--- Analyzing Performance for {agent.__class__.__name__} ---")
    
    # Plot learning curve (rewards per episode)
    plt.figure(figsize=(VisualizationConfig.FIGURE_SIZE[0], VisualizationConfig.FIGURE_SIZE[1] / 2))
    plt.plot(agent.episode_rewards)
    plt.title(f"Learning Curve: {agent.__class__.__name__}", fontsize=VisualizationConfig.FONT_SIZE + 2)
    plt.xlabel("Episode", fontsize=VisualizationConfig.FONT_SIZE)
    plt.ylabel("Total Reward", fontsize=VisualizationConfig.FONT_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filepath_prefix}_learning_curve.png")
    plt.close()
    print(f"  Saved learning curve to {save_dir}/{filepath_prefix}_learning_curve.png")

    # Plot episode steps
    plt.figure(figsize=(VisualizationConfig.FIGURE_SIZE[0], VisualizationConfig.FIGURE_SIZE[1] / 2))
    plt.plot(agent.episode_steps)
    plt.title(f"Episode Steps: {agent.__class__.__name__}", fontsize=VisualizationConfig.FONT_SIZE + 2)
    plt.xlabel("Episode", fontsize=VisualizationConfig.FONT_SIZE)
    plt.ylabel("Steps", fontsize=VisualizationConfig.FONT_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filepath_prefix}_episode_steps.png")
    plt.close()
    print(f"  Saved episode steps to {save_dir}/{filepath_prefix}_episode_steps.png")

    summary = {
        "total_episodes": len(agent.episode_rewards),
        "mean_final_reward": np.mean(agent.episode_rewards[-ExperimentConfig.EVAL_EPISODES:]), # Avg of last N episodes
        "mean_total_reward": np.mean(agent.episode_rewards),
        "std_total_reward": np.std(agent.episode_rewards),
        "mean_total_steps": np.mean(agent.episode_steps),
        "std_total_steps": np.std(agent.episode_steps),
    }
    print("  Performance Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    • {key}: {value:.2f}")
        else:
            print(f"    • {key}: {value}")

    print("Performance analysis completed.")
    return summary


if __name__ == "__main__":
    # Example Usage
    env = GridWorld()
    print("--- Evaluation Module Demo ---")

    # Train a Q-Learning agent for demonstration
    q_agent = QLearningAgent(env, num_episodes=300)
    q_agent.train(num_episodes=300, print_every=100)
    
    # Evaluate the Q-Learning agent
    q_eval_results = evaluate_agent(q_agent, env, save_results=False)
    print(f"Q-Learning Evaluation: Avg Reward = {q_eval_results['avg_reward']:.2f}, Success Rate = {q_eval_results['success_rate']*100:.1f}%")

    # Train a SARSA agent for comparison
    sarsa_agent = SARSAAgent(env, num_episodes=300)
    sarsa_agent.train(num_episodes=300, print_every=100)

    # Compare Q-Learning and SARSA
    agents_to_compare = {
        "Q-Learning": q_agent,
        "SARSA": sarsa_agent,
    }
    comparison_results = compare_agents(
        agents_to_compare,
        env,
        filepath=f"{VisualizationConfig.SAVE_DIR}/agent_comparison_boxplot.png"
    )
    print("Agent comparison plot saved to visualizations/agent_comparison_boxplot.png")

    # Analyze Q-Learning performance
    analysis_summary = analyze_performance(q_agent, filepath_prefix="q_learning_agent")
    print("Performance analysis plots saved.")
    print("\nAll evaluation examples finished.")
