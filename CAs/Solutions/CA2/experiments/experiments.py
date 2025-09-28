"""
Experiment Functions for Reinforcement Learning

This module contains functions for running various experiments
with different parameters and environments.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd

from ..environments import GridWorld, create_custom_environment
from ..agents.policies import (
    RandomPolicy,
    GreedyPolicy,
    CustomPolicy,
    GreedyActionPolicy,
)
from ..agents.algorithms import (
    policy_evaluation,
    policy_iteration,
    value_iteration,
    q_learning,
)
from ..utils.visualization import plot_value_function, plot_policy, plot_learning_curve

np.random.seed(42)


def experiment_discount_factors(env, policy, gamma_values=[0.1, 0.5, 0.9, 0.99]):
    """
    Experiment with different discount factors.

    Args:
        env: GridWorld environment
        policy: Policy to evaluate
        gamma_values: List of discount factors to test
    """
    print("=== Experiment: Effect of Discount Factor ===")

    results = {}

    for gamma in gamma_values:
        print(f"\nEvaluating policy with gamma = {gamma}")
        values = policy_evaluation(env, policy, gamma=gamma)
        results[gamma] = values

        print(f"Value of start state (0,0): {values[(0,0)]:.3f}")
        print(f"Value of state near goal (2,2): {values[(2,2)]:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, gamma in enumerate(gamma_values):
        ax = axes[idx]
        values = results[gamma]

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
        ax.set_title(f"γ = {gamma}", fontsize=12, fontweight="bold")

        for i in range(env.size):
            for j in range(env.size):
                state = (i, j)
                if state not in env.obstacles and state != env.goal_state:
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

        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))

    fig.suptitle(
        "Value Functions for Different Discount Factors", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    return results


def experiment_policy_comparison(env, gamma=0.9):
    """
    Compare different policies.

    Args:
        env: GridWorld environment
        gamma: Discount factor
    """
    print("=== Experiment: Policy Comparison ===")

    policies = [
        RandomPolicy(env),
        CustomPolicy(env),
        GreedyActionPolicy(env, gamma=gamma),
    ]

    policy_names = ["Random Policy", "Custom Policy", "Optimal Policy"]

    optimal_values = policy_evaluation(env, policies[2], gamma=gamma)
    policies[2] = GreedyActionPolicy(env, optimal_values, gamma=gamma)

    from ..utils.visualization import compare_policies

    compare_policies(env, policies, policy_names, gamma)

    for policy, name in zip(policies, policy_names):
        values = policy_evaluation(env, policy, gamma=gamma)
        start_value = values[(0, 0)]
        print(f"{name}: Start state value = {start_value:.3f}")


def experiment_policy_iteration(env, gamma=0.9):
    """
    Run policy iteration and show the process.

    Args:
        env: GridWorld environment
        gamma: Discount factor
    """
    print("=== Experiment: Policy Iteration ===")

    optimal_policy, optimal_values, history = policy_iteration(env, gamma=gamma)

    print(f"\nPolicy iteration completed in {len(history)} iterations")
    print(f"Final value of start state: {optimal_values[(0,0)]:.3f}")

    plot_value_function(env, optimal_values, "Optimal Value Function")
    plot_policy(env, optimal_policy, "Optimal Policy")

    return optimal_policy, optimal_values, history


def experiment_value_iteration(env, gamma=0.9):
    """
    Run value iteration and show convergence.

    Args:
        env: GridWorld environment
        gamma: Discount factor
    """
    print("=== Experiment: Value Iteration ===")

    optimal_values, optimal_policy, history = value_iteration(env, gamma=gamma)

    print(f"\nValue iteration completed in {len(history)} iterations")
    print(f"Final value of start state: {optimal_values[(0,0)]:.3f}")

    from ..utils.visualization import plot_value_iteration_convergence

    plot_value_iteration_convergence([h["values"] for h in history])

    plot_value_function(env, optimal_values, "Value Iteration - Final Values")
    plot_policy(env, optimal_policy, "Value Iteration - Optimal Policy")

    return optimal_values, optimal_policy, history


def experiment_q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Run Q-learning experiment.

    Args:
        env: GridWorld environment
        num_episodes: Number of episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
    """
    print("=== Experiment: Q-Learning ===")

    Q, episode_rewards = q_learning(env, num_episodes, alpha, gamma, epsilon)

    print(f"Q-learning completed with {num_episodes} episodes")
    print(".3f")

    plot_learning_curve(episode_rewards, "Q-Learning Learning Curve")

    from ..agents.algorithms import compute_v_from_q

    values = compute_v_from_q(Q, env)
    policy = GreedyPolicy(env, Q)

    plot_value_function(env, values, "Q-Learning - Value Function")
    plot_policy(env, policy, "Q-Learning - Learned Policy")

    return Q, values, policy, episode_rewards


def experiment_environment_modifications():
    """
    Experiment with different environment configurations.
    """
    print("=== Experiment: Environment Modifications ===")

    configs = [
        {"obstacles": [(1, 1), (2, 1), (1, 2)], "name": "Standard"},
        {"obstacles": [(1, 1)], "name": "Easy (Few Obstacles)"},
        {
            "obstacles": [(1, 1), (2, 1), (1, 2), (2, 2)],
            "name": "Hard (Many Obstacles)",
        },
        {"obstacles": [], "name": "No Obstacles"},
    ]

    results = {}

    for config in configs:
        print(f"\nTesting environment: {config['name']}")
        env = create_custom_environment(obstacles=config["obstacles"])

        optimal_policy, optimal_values, _ = policy_iteration(
            env, gamma=0.9, max_iterations=10
        )

        start_value = optimal_values[(0, 0)]
        results[config["name"]] = start_value

        print(f"Start state value: {start_value:.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = list(results.keys())
    values = list(results.values())

    bars = ax.bar(names, values, color=["blue", "green", "red", "orange"])
    ax.set_ylabel("Value of Start State")
    ax.set_title("Environment Difficulty Comparison", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return results


def run_all_experiments(env=None):
    """
    Run all experiments in sequence.

    Args:
        env: GridWorld environment (creates default if None)
    """
    if env is None:
        env = GridWorld()

    print("Starting comprehensive RL experiments...\n")

    random_policy = RandomPolicy(env)
    experiment_discount_factors(env, random_policy)

    experiment_policy_comparison(env)

    experiment_policy_iteration(env)

    experiment_value_iteration(env)

    experiment_q_learning(env)

    experiment_environment_modifications()

    print("\nAll experiments completed!")


if __name__ == "__main__":

    run_all_experiments()
