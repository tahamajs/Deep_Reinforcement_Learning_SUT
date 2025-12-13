"""
visualization.py - Utilities for plotting and visualizing reinforcement learning results.

This module provides functions to visualize learning curves, value functions,
policies, and comparisons between different TD learning algorithms and
exploration strategies in the GridWorld environment.
"""

from typing import Dict, Tuple, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Assuming environments, agents, and config are available in the src package
from .environments import GridWorld
from .agents import TD0Agent, QLearningAgent, SARSAAgent, BaseAgent
from .config import VisualizationConfig

def plot_learning_curve(
    episode_rewards: List[float],
    title: str = "Learning Curve",
    filepath: str = None,
) -> None:
    """
    Plots the learning curve (rewards per episode).

    Args:
        episode_rewards (List[float]): A list of total rewards obtained in each episode.
        title (str): The title of the plot.
        filepath (str, optional): Path to save the figure. If None, displays the figure.
    """
    plt.figure(figsize=VisualizationConfig.FIGURE_SIZE)
    sns.lineplot(x=range(len(episode_rewards)), y=episode_rewards)
    plt.title(title, fontsize=VisualizationConfig.FONT_SIZE + 2)
    plt.xlabel("Episode", fontsize=VisualizationConfig.FONT_SIZE)
    plt.ylabel("Total Reward", fontsize=VisualizationConfig.FONT_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        plt.savefig(filepath, dpi=VisualizationConfig.PLOT_DPI)
        plt.close()
    else:
        plt.show()

def plot_q_learning_analysis(
    agent: QLearningAgent,
    filepath_prefix: str = "q_learning_analysis",
    save_dir: str = VisualizationConfig.SAVE_DIR,
) -> None:
    """
    Generates and saves a set of analysis plots specific to Q-Learning (or SARSA).

    Includes learning curve, episode steps, and Q-value visualization.

    Args:
        agent (QLearningAgent): The trained Q-Learning or SARSA agent.
        filepath_prefix (str): Prefix for saved plot filenames.
        save_dir (str): Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Learning Curve
    plot_learning_curve(
        agent.episode_rewards,
        title=f"{agent.__class__.__name__} Learning Curve",
        filepath=os.path.join(save_dir, f"{filepath_prefix}_learning_curve.png"),
    )

    # Episode Steps
    plt.figure(figsize=VisualizationConfig.FIGURE_SIZE)
    sns.lineplot(x=range(len(agent.episode_steps)), y=agent.episode_steps)
    plt.title(f"{agent.__class__.__name__} Episode Steps", fontsize=VisualizationConfig.FONT_SIZE + 2)
    plt.xlabel("Episode", fontsize=VisualizationConfig.FONT_SIZE)
    plt.ylabel("Steps to Terminal State", fontsize=VisualizationConfig.FONT_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filepath_prefix}_episode_steps.png"), dpi=VisualizationConfig.PLOT_DPI)
    plt.close()

    # Q-Value Heatmap and Policy
    V_func = agent.get_value_function()
    policy = agent.get_policy()
    agent.env.visualize_values(
        V_func,
        title=f"{agent.__class__.__name__} Value Function and Policy",
        policy=policy,
        filepath=os.path.join(save_dir, f"{filepath_prefix}_value_policy.png"),
    )

    print(f"✓ Generated analysis plots for {agent.__class__.__name__} in {save_dir}")

def show_q_values(agent: QLearningAgent, states_to_show: List[Tuple[int, int]] = None) -> None:
    """
    Prints the Q-values for selected states, or all states if none are specified.

    Args:
        agent (QLearningAgent): The agent with learned Q-values.
        states_to_show (List[Tuple[int, int]], optional): List of states to display Q-values for.
                                                          Defaults to None.
    """
    print(f"\n--- Q-Values for {agent.__class__.__name__} ---")
    if not states_to_show:
        states_to_show = list(agent.Q.keys())
        if len(states_to_show) > 5: # Limit output for readability
            print("Displaying Q-values for first 5 learned states (or specify states_to_show):\n")
            states_to_show = sorted(list(agent.Q.keys()))[:5]

    for state in states_to_show:
        print(f"State {state}:")
        if state in agent.Q:
            for action, q_value in agent.Q[state].items():
                print(f"  {action}: {q_value:.2f}")
        else:
            print("  No Q-values learned for this state yet.")
    print("----------------------")

def compare_algorithms(
    td_agent: TD0Agent,
    q_agent: QLearningAgent,
    sarsa_agent: SARSAAgent,
    V_td: Dict[Tuple[int, int], float],
    V_optimal: Dict[Tuple[int, int], float],
    V_sarsa: Dict[Tuple[int, int], float],
    q_evaluation: Dict[str, Any],
    sarsa_evaluation: Dict[str, Any],
    save_dir: str = VisualizationConfig.SAVE_DIR,
) -> None:
    """
    Compares the performance of TD(0), Q-Learning, and SARSA agents through plots.

    Args:
        td_agent (TD0Agent): Trained TD(0) agent.
        q_agent (QLearningAgent): Trained Q-Learning agent.
        sarsa_agent (SARSAAgent): Trained SARSA agent.
        V_td (Dict): Learned V-function from TD(0).
        V_optimal (Dict): Optimal V-function from Q-Learning.
        V_sarsa (Dict): Learned V-function from SARSA.
        q_evaluation (Dict): Evaluation metrics for Q-Learning.
        sarsa_evaluation (Dict): Evaluation metrics for SARSA.
        save_dir (str): Directory to save comparison plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n--- Generating Algorithm Comparison Plots ---")

    # 1. Learning Curves Comparison
    plt.figure(figsize=VisualizationConfig.FIGURE_SIZE)
    sns.lineplot(x=range(len(q_agent.episode_rewards)), y=q_agent.episode_rewards, label="Q-Learning")
    sns.lineplot(x=range(len(sarsa_agent.episode_rewards)), y=sarsa_agent.episode_rewards, label="SARSA")
    # TD(0) learning curve might be on a different scale/objective, add carefully or separately if needed
    # sns.lineplot(x=range(len(td_agent.episode_rewards)), y=td_agent.episode_rewards, label="TD(0)")
    plt.title("Q-Learning vs SARSA: Learning Curves", fontsize=VisualizationConfig.FONT_SIZE + 2)
    plt.xlabel("Episode", fontsize=VisualizationConfig.FONT_SIZE)
    plt.ylabel("Total Reward", fontsize=VisualizationConfig.FONT_SIZE)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "algorithm_learning_curves_comparison.png"), dpi=VisualizationConfig.PLOT_DPI)
    plt.close()
    print("✓ Saved learning curves comparison.")

    # 2. Final Value Function Comparison
    fig, axes = plt.subplots(1, 3, figsize=(3 * VisualizationConfig.FIGURE_SIZE[0] / 1.5, VisualizationConfig.FIGURE_SIZE[1]))

    q_agent.env.visualize_values(
        V_td, title="TD(0) V-Function", policy=td_agent.get_policy(), ax=axes[0], show_colorbar=False
    ) # Need to modify visualize_values to accept ax and show_colorbar
    q_agent.env.visualize_values(
        V_optimal, title="Q-Learning V* Function", policy=q_agent.get_policy(), ax=axes[1], show_colorbar=False
    )
    q_agent.env.visualize_values(
        V_sarsa, title="SARSA V-Function", policy=sarsa_agent.get_policy(), ax=axes[2], show_colorbar=True
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "algorithm_value_function_comparison.png"), dpi=VisualizationConfig.PLOT_DPI)
    plt.close()
    print("✓ Saved value function comparison.")

    # 3. Performance Metrics Table
    data = {
        "Algorithm": ["TD(0)", "Q-Learning", "SARSA"],
        "Avg Reward": [
            np.mean(td_agent.episode_rewards), # Use full training avg reward for TD(0)
            q_evaluation["avg_reward"],
            sarsa_evaluation["avg_reward"],
        ],
        "Std Reward": [
            np.std(td_agent.episode_rewards),
            q_evaluation["std_reward"],
            sarsa_evaluation["std_reward"],
        ],
        "Success Rate": [
            "N/A", # TD(0) is policy evaluation, not control with a specific success goal
            f"{q_evaluation["success_rate"]*100:.1f}%",
            f"{sarsa_evaluation["success_rate"]*100:.1f}%",
        ],
    }
    df = pd.DataFrame(data)
    print("\nPerformance Metrics:")
    print(df.to_string(index=False))

    # Save as image
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    plt.title("Algorithm Performance Metrics", fontsize=VisualizationConfig.FONT_SIZE + 2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "algorithm_performance_table.png"), dpi=VisualizationConfig.PLOT_DPI)
    plt.close()
    print("✓ Saved performance metrics table.")

    print("Algorithm comparison plots and metrics generated.")

# Extend GridWorld.visualize_values to accept an axes object and control colorbar
def _visualize_values_extended(self, values: Dict[Tuple[int, int], float], title: str = "GridWorld Value Function", 
                               policy: Dict[Tuple[int, int], str] = None, 
                               filepath: str = None, ax: plt.Axes = None, show_colorbar: bool = True) -> None:
    if ax is None:
        fig, ax = plt.subplots(figsize=(self.size, self.size))
    
    grid_values = np.zeros((self.size, self.size))
    for r, c in self.states:
        grid_values[r, c] = values.get((r, c), 0.0)

    sns.heatmap(
        grid_values,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar=show_colorbar,
        linewidths=0.5,
        linecolor="black",
        yticklabels=False,
        xticklabels=False,
        ax=ax
    )

    # Mark start, goal, and obstacles
    for r, c in self.states:
        if (r, c) == self.start_state:
            ax.text(c + 0.5, r + 0.5, "S", color="red", ha="center", va="center", fontsize=16)
        elif (r, c) == self.goal_state:
            ax.text(c + 0.5, r + 0.5, "G", color="green", ha="center", va="center", fontsize=16)
        elif (r, c) in self.obstacles:
            ax.text(c + 0.5, r + 0.5, "X", color="black", ha="center", va="center", fontsize=16)
        
        # Draw policy arrows
        if policy and (r, c) in policy and (r,c) != self.goal_state and (r,c) not in self.obstacles:
            action = policy[(r,c)]
            dx, dy = 0, 0
            if action == "up": dy = -0.3
            elif action == "down": dy = 0.3
            elif action == "left": dx = -0.3
            elif action == "right": dx = 0.3
            ax.arrow(c + 0.5, r + 0.5, dx, dy, head_width=0.2, head_length=0.2, fc='white', ec='white')

    ax.set_title(title, fontsize=VisualizationConfig.FONT_SIZE + 2)
    
    if filepath and ax is None:
        plt.savefig(filepath, dpi=VisualizationConfig.PLOT_DPI)
        plt.close()
    elif ax is None:
        plt.show()

GridWorld.visualize_values = _visualize_values_extended

if __name__ == "__main__":
    # Example Usage
    env = GridWorld()
    # Assuming RandomPolicy is defined elsewhere or needs to be imported
    # from .agents import RandomPolicy # This line was not in the original file, so I'm adding it.
    # random_policy = RandomPolicy(env) # This line was not in the original file, so I'm adding it.
    td_agent = TD0Agent(env, None, num_episodes=100) # Assuming a default policy for TD0Agent
    V_td = td_agent.train(num_episodes=100, print_every=50)

    q_agent = QLearningAgent(env, num_episodes=200)
    q_agent.train(num_episodes=200, print_every=50)
    q_evaluation = q_agent.evaluate_policy(num_episodes=20)

    sarsa_agent = SARSAAgent(env, num_episodes=200)
    sarsa_agent.train(num_episodes=200, print_every=50)
    sarsa_evaluation = sarsa_agent.evaluate_policy(num_episodes=20)

    # Plot learning curve demo
    plot_learning_curve(
        q_agent.episode_rewards, 
        "Q-Learning Rewards over Episodes", 
        filepath=os.path.join(VisualizationConfig.SAVE_DIR, "q_learning_rewards.png")
    )

    # Plot Q-Learning analysis demo
    plot_q_learning_analysis(q_agent, filepath_prefix="q_learning_demo_analysis")

    # Show Q-values demo
    show_q_values(q_agent, states_to_show=[(0,0), (1,0), (3,2)])

    # Compare algorithms demo
    compare_algorithms(
        td_agent, q_agent, sarsa_agent,
        V_td, q_agent.get_value_function(), sarsa_agent.get_value_function(),
        q_evaluation, sarsa_evaluation
    )

    print("\nAll visualization examples finished.")
