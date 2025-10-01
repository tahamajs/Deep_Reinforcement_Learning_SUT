"""
Comprehensive demonstration script for CA13 advanced RL methods.
Run this to see all techniques in action.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(".."))

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from CA13 import (
    DQNAgent,
    ModelBasedAgent,
    SampleEfficientAgent,
    OptionsCriticAgent,
    FeudalAgent,
    AdvancedRLEvaluator,
    IntegratedAdvancedAgent,
    set_seed,
    get_device,
    train_dqn_agent,
    train_model_based_agent,
    evaluate_agent,
)

from environments.grid_world import SimpleGridWorld


def main():
    """Run comprehensive demonstration."""
    print("=" * 80)
    print("CA13: COMPREHENSIVE ADVANCED RL DEMONSTRATION")
    print("=" * 80)

    # Setup
    seed = 42
    set_seed(seed)
    device = get_device()
    
    print(f"\n✓ Device: {device}")
    print(f"✓ Random seed: {seed}")

    # Create environment
    try:
        env = gym.make("CartPole-v1")
        env_name = "CartPole-v1"
    except:
        env = SimpleGridWorld(size=5)
        env_name = "SimpleGridWorld-5x5"

    state_dim = (
        env.observation_space.shape[0]
        if hasattr(env.observation_space, "shape")
        else 2
    )
    action_dim = env.action_space.n

    print(f"\n✓ Environment: {env_name}")
    print(f"✓ State dim: {state_dim}, Action dim: {action_dim}")

    # Initialize agents
    print("\n" + "=" * 80)
    print("INITIALIZING AGENTS")
    print("=" * 80)

    agents_config = {
        "DQN": DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=1e-3,
        ),
        "Model-Based": ModelBasedAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=1e-3,
        ),
        "Sample-Efficient": SampleEfficientAgent(
            state_dim=state_dim, action_dim=action_dim, lr=1e-3
        ),
    }

    for name in agents_config.keys():
        print(f"✓ {name}")

    # Training
    print("\n" + "=" * 80)
    print("TRAINING AGENTS")
    print("=" * 80)

    num_episodes = 100
    results = {}

    # Train DQN
    print("\n1. Training DQN...")
    dqn_results = train_dqn_agent(
        env=gym.make(env_name) if "CartPole" in env_name else SimpleGridWorld(size=5),
        agent=agents_config["DQN"],
        num_episodes=num_episodes,
        max_steps=500,
        eval_interval=20,
    )
    results["DQN"] = dqn_results
    print(
        f"   Final reward: {np.mean(dqn_results['rewards'][-10:]):.2f}"
    )

    # Train Model-Based
    print("\n2. Training Model-Based Agent...")
    mb_results = train_model_based_agent(
        env=gym.make(env_name) if "CartPole" in env_name else SimpleGridWorld(size=5),
        agent=agents_config["Model-Based"],
        num_episodes=num_episodes,
        max_steps=500,
        eval_interval=20,
        planning_steps=10,
    )
    results["Model-Based"] = mb_results
    print(
        f"   Final reward: {np.mean(mb_results['rewards'][-10:]):.2f}"
    )

    # Evaluation
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    for name, agent in agents_config.items():
        eval_env = (
            gym.make(env_name) if "CartPole" in env_name else SimpleGridWorld(size=5)
        )
        eval_results = evaluate_agent(eval_env, agent, num_episodes=10)
        print(f"\n{name}:")
        print(
            f"  Mean Return: {eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}"
        )
        print(
            f"  Mean Length: {eval_results['mean_length']:.2f} ± {eval_results['std_length']:.2f}"
        )
        results[name]["evaluation"] = eval_results

    # Visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    ax = axes[0]
    for name, result in results.items():
        rewards = result["rewards"]
        smoothed = pd.Series(rewards).rolling(window=10, min_periods=1).mean()
        ax.plot(smoothed, label=name, linewidth=2)

    ax.set_title("Learning Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(alpha=0.3)

    # Performance comparison
    ax = axes[1]
    names = list(results.keys())
    final_performance = [np.mean(results[name]["rewards"][-10:]) for name in names]

    bars = ax.bar(names, final_performance, alpha=0.7)
    ax.set_title("Final Performance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Return (last 10 episodes)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("../results/comprehensive_demo.png", dpi=150, bbox_inches="tight")
    print("✓ Saved visualization to results/comprehensive_demo.png")

    plt.show()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
