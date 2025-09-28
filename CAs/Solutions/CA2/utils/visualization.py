"""
Visualization Functions for Reinforcement Learning

This module contains functions for visualizing value functions, policies,
and learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def plot_value_function(env, values, title="Value Function", figsize=(8, 6)):
    """
    Plot the value function as a heatmap.

    Args:
        env: GridWorld environment
        values: Dictionary mapping states to values
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    grid = np.zeros((env.size, env.size))

    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if state in env.obstacles:
                grid[i, j] = -1
            elif state == env.goal_state:
                grid[i, j] = env.goal_reward
            else:
                grid[i, j] = values.get(state, 0)

    im = ax.imshow(grid, cmap="RdYlGn", aspect="equal")

    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if state == env.goal_state:
                ax.text(
                    j,
                    i,
                    "G",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color="white",
                )
            elif state in env.obstacles:
                ax.text(
                    j,
                    i,
                    "X",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color="white",
                )
            elif state == env.start_state:
                ax.text(
                    j,
                    i,
                    f"S\n{values.get(state, 0):.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )
            else:
                ax.text(
                    j,
                    i,
                    f"{values.get(state, 0):.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_policy(env, policy, title="Policy", figsize=(8, 6)):
    """
    Visualize a policy with arrows indicating action preferences.

    Args:
        env: GridWorld environment
        policy: Policy object
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    grid = np.zeros((env.size, env.size))

    for i, j in env.obstacles:
        grid[i, j] = -1

    goal_i, goal_j = env.goal_state
    grid[goal_i, goal_j] = 1

    im = ax.imshow(grid, cmap="RdYlGn", aspect="equal")

    arrow_map = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if state not in env.obstacles and state != env.goal_state:
                action = policy.get_action(state)
                if action and action in arrow_map:
                    ax.text(
                        j,
                        i,
                        arrow_map[action],
                        ha="center",
                        va="center",
                        fontsize=20,
                        fontweight="bold",
                        color="blue",
                    )

    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if state == env.goal_state:
                ax.text(
                    j,
                    i - 0.3,
                    "G",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color="white",
                )
            elif state in env.obstacles:
                ax.text(
                    j,
                    i - 0.3,
                    "X",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color="white",
                )
            elif state == env.start_state:
                ax.text(
                    j,
                    i - 0.3,
                    "S",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black",
                )

    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_q_values(env, Q, title="Q-Values", figsize=(12, 8)):
    """
    Plot Q-values for each state-action pair.

    Args:
        env: GridWorld environment
        Q: Dictionary mapping (state, action) to Q-values
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    action_names = ["up", "down", "left", "right"]

    for idx, action in enumerate(action_names):
        ax = axes[idx]

        grid = np.zeros((env.size, env.size))

        for i in range(env.size):
            for j in range(env.size):
                state = (i, j)
                if state in env.obstacles or state == env.goal_state:
                    grid[i, j] = 0
                else:
                    grid[i, j] = Q.get((state, action), 0)

        im = ax.imshow(grid, cmap="RdYlGn", aspect="equal")

        for i in range(env.size):
            for j in range(env.size):
                state = (i, j)
                if state not in env.obstacles and state != env.goal_state:
                    q_val = Q.get((state, action), 0)
                    ax.text(
                        j,
                        i,
                        f"{q_val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )

        ax.set_title(f"Q(s,{action})", fontsize=12, fontweight="bold")
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(rewards, title="Learning Curve", window=50):
    """
    Plot learning curve with moving average.

    Args:
        rewards: List of episode rewards
        title: Plot title
        window: Window size for moving average
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    episodes = range(1, len(rewards) + 1)
    ax.plot(episodes, rewards, alpha=0.3, color="blue", label="Episode Reward")

    if len(rewards) >= window:
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        ax.plot(
            episodes,
            moving_avg,
            color="red",
            linewidth=2,
            label=f"Moving Average ({window} episodes)",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_value_iteration_convergence(
    iteration_history, title="Value Iteration Convergence"
):
    """
    Plot how values change during value iteration.

    Args:
        iteration_history: List of value functions from each iteration
        title: Plot title
    """
    if not iteration_history:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    max_values = [max(v.values()) for v in iteration_history]
    axes[0].plot(range(1, len(max_values) + 1), max_values, "b-o")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Max Value")
    axes[0].set_title("Maximum Value vs Iteration")
    axes[0].grid(True, alpha=0.3)

    key_states = [(0, 0), (1, 0), (2, 2), (3, 2)]
    state_labels = ["(0,0)", "(1,0)", "(2,2)", "(3,2)"]

    for state, label in zip(key_states, state_labels):
        values_over_time = [v.get(state, 0) for v in iteration_history]
        axes[1].plot(
            range(1, len(values_over_time) + 1),
            values_over_time,
            "o-",
            label=label,
            markersize=4,
        )

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Value Changes for Key States")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    env_size = 4
    final_values = iteration_history[-1]
    grid = np.zeros((env_size, env_size))

    for i in range(env_size):
        for j in range(env_size):
            state = (i, j)
            grid[i, j] = final_values.get(state, 0)

    im = axes[2].imshow(grid, cmap="RdYlGn", aspect="equal")
    axes[2].set_title("Final Value Function")
    axes[2].set_xticks(range(env_size))
    axes[2].set_yticks(range(env_size))
    plt.colorbar(im, ax=axes[2])

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def compare_policies(env, policies, policy_names, gamma=0.9, title="Policy Comparison"):
    """
    Compare different policies by evaluating their value functions.

    Args:
        env: GridWorld environment
        policies: List of policy objects
        policy_names: List of policy names
        gamma: Discount factor
        title: Plot title
    """
    from ..agents.algorithms import policy_evaluation

    fig, axes = plt.subplots(1, len(policies), figsize=(5 * len(policies), 4))

    if len(policies) == 1:
        axes = [axes]

    for idx, (policy, name) in enumerate(zip(policies, policy_names)):
        ax = axes[idx]

        values = policy_evaluation(env, policy, gamma)

        grid = np.zeros((env.size, env.size))
        for i in range(env.size):
            for j in range(env.size):
                state = (i, j)
                if state in env.obstacles:
                    grid[i, j] = -1
                elif state == env.goal_state:
                    grid[i, j] = env.goal_reward
                else:
                    grid[i, j] = values.get(state, 0)

        im = ax.imshow(grid, cmap="RdYlGn", aspect="equal")

        for i in range(env.size):
            for j in range(env.size):
                state = (i, j)
                if state == env.goal_state:
                    ax.text(
                        j,
                        i,
                        "G",
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                        color="white",
                    )
                elif state in env.obstacles:
                    ax.text(
                        j,
                        i,
                        "X",
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                        color="white",
                    )
                else:
                    val = values.get(state, 0)
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )

        ax.set_title(f"{name} Policy", fontsize=12, fontweight="bold")
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
