"""
Basic DQN Experiment for CA07
=============================
This script runs a basic DQN experiment on CartPole-v1 environment
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from training_examples import DQNAgent, train_dqn_agent
import warnings

warnings.filterwarnings("ignore")


def main():
    """Run basic DQN experiment"""
    print("Basic DQN Experiment")
    print("=" * 30)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create environment
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    print(f"Environment: {env_name}")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Train DQN agent
    print("\nTraining DQN agent...")
    result = train_dqn_agent(
        DQNAgent,
        env_name=env_name,
        episodes=300,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        replay_buffer_size=10000,
        batch_size=64,
        target_update_freq=10,
    )

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # Learning curve
    plt.subplot(2, 3, 1)
    scores = result["scores"]
    smoothed_scores = np.convolve(scores, np.ones(20) / 20, mode="valid")
    plt.plot(scores, alpha=0.3, color="blue", label="Raw scores")
    plt.plot(smoothed_scores, color="blue", linewidth=2, label="Smoothed scores")
    plt.title("DQN Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss curve
    plt.subplot(2, 3, 2)
    losses = result["losses"]
    smoothed_losses = np.convolve(losses, np.ones(20) / 20, mode="valid")
    plt.plot(losses, alpha=0.3, color="red", label="Raw losses")
    plt.plot(smoothed_losses, color="red", linewidth=2, label="Smoothed losses")
    plt.title("DQN Loss Curve")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Epsilon decay
    plt.subplot(2, 3, 3)
    epsilon_history = result["epsilon_history"]
    plt.plot(epsilon_history, color="green", linewidth=2)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.3)

    # Score distribution
    plt.subplot(2, 3, 4)
    plt.hist(scores, bins=30, alpha=0.7, color="purple", edgecolor="black")
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Recent performance
    plt.subplot(2, 3, 5)
    recent_scores = scores[-100:]
    plt.plot(recent_scores, color="orange", linewidth=2)
    plt.title("Recent Performance (Last 100 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)

    # Performance statistics
    plt.subplot(2, 3, 6)
    stats = {
        "Mean Score": np.mean(scores),
        "Max Score": np.max(scores),
        "Min Score": np.min(scores),
        "Std Score": np.std(scores),
        "Final 50 Avg": np.mean(scores[-50:]),
        "Episodes > 200": np.sum(np.array(scores) > 200),
    }

    y_pos = np.arange(len(stats))
    plt.barh(y_pos, list(stats.values()), alpha=0.7, color="cyan")
    plt.yticks(y_pos, list(stats.keys()))
    plt.title("Performance Statistics")
    plt.xlabel("Value")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/basic_dqn_experiment.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print results
    print("\nExperiment Results:")
    print("=" * 20)
    print(f"Total episodes: {len(scores)}")
    print(f"Mean score: {np.mean(scores):.2f}")
    print(f"Max score: {np.max(scores):.2f}")
    print(f"Min score: {np.min(scores):.2f}")
    print(f"Final 50 episodes average: {np.mean(scores[-50:]):.2f}")
    print(f"Episodes with score > 200: {np.sum(np.array(scores) > 200)}")

    # Evaluate final performance
    print("\nEvaluating final performance...")
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=0.01,  # Low exploration for evaluation
        epsilon_end=0.01,
        epsilon_decay=1.0,
    )

    # Load the trained weights (simplified - in practice you'd save/load properly)
    evaluation_results = agent.evaluate(env, num_episodes=10)

    print(f"Evaluation results (10 episodes):")
    print(f"  Mean reward: {evaluation_results['mean_reward']:.2f}")
    print(f"  Std reward: {evaluation_results['std_reward']:.2f}")
    print(f"  Max reward: {evaluation_results['max_reward']:.2f}")
    print(f"  Min reward: {evaluation_results['min_reward']:.2f}")

    env.close()
    print("\nBasic DQN experiment completed successfully!")


if __name__ == "__main__":
    main()
