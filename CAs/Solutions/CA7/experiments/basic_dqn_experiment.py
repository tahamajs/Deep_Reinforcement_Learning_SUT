"""
Basic DQN Experiment
===================

This script demonstrates the basic DQN implementation on the CartPole environment.
It includes training, evaluation, and performance analysis.

Usage:
    python experiments/basic_dqn_experiment.py

Author: CA7 Implementation
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..agents.core import DQNAgent, PerformanceAnalyzer
import warnings

warnings.filterwarnings("ignore")


def run_basic_dqn_experiment():
    """
    Run the basic DQN experiment on CartPole-v1

    Returns:
        Trained agent and training results
    """
    print("=" * 60)
    print("Basic DQN Experiment - CartPole-v1")
    print("=" * 60)

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
        buffer_size=20000,  # Experience replay buffer size
        batch_size=64,  # Training batch size
        target_update_freq=100,  # Target network update frequency
    )

    num_episodes = 300
    max_steps_per_episode = 500

    print("Training Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    print(f"  Learning rate: {agent.optimizer.param_groups[0]['lr']}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Epsilon decay: {agent.epsilon_decay}")
    print(f"  Buffer size: {len(agent.replay_buffer.capacity)}")
    print(f"  Batch size: {agent.batch_size}")
    print()

    print("Starting training...")
    print("-" * 60)

    episode_rewards = []
    best_reward = 0
    solved_episode = None

    for episode in range(num_episodes):
        reward, steps = agent.train_episode(env, max_steps=max_steps_per_episode)
        episode_rewards.append(reward)

        if reward > best_reward:
            best_reward = reward

        if len(episode_rewards) >= 100:
            avg_last_100 = np.mean(episode_rewards[-100:])
            if avg_last_100 >= 195 and solved_episode is None:
                solved_episode = episode + 1
                print(f"🎉 Environment solved at episode {solved_episode}!")

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            recent_avg = (
                np.mean(episode_rewards[-20:])
                if len(episode_rewards) >= 20
                else avg_reward
            )
            eval_results = agent.evaluate(env, num_episodes=5)

            print(
                f"Episode {episode+1:3d} | "
                f"Train Reward: {reward:6.1f} | "
                f"Avg (50): {avg_reward:6.1f} | "
                f"Eval: {eval_results['mean_reward']:6.1f} ± {eval_results['std_reward']:4.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.replay_buffer)}"
            )

    print("-" * 60)
    print("Training completed!")
    print()

    print("Final Evaluation:")
    print("-" * 30)

    final_eval = agent.evaluate(env, num_episodes=20)
    print(
        f"Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}"
    )
    print(f"Min Reward: {final_eval['min_reward']:.2f}")
    print(f"Max Reward: {final_eval['max_reward']:.2f}")
    print(
        f"Success Rate (>195): {(np.array(final_eval['mean_reward'] > 195)).mean():.1%}"
    )
    print()

    print("Training Summary:")
    print("-" * 30)
    print(f"Total episodes: {num_episodes}")
    print(f"Best episode reward: {best_reward}")
    print(f"Final average reward: {np.mean(episode_rewards[-50:]):.1f}")
    if solved_episode:
        print(f"Environment solved at episode: {solved_episode}")
    else:
        print("Environment not fully solved (avg < 195)")
    print()

    results = {
        "agent": agent,
        "rewards": episode_rewards,
        "losses": agent.losses,
        "epsilon_history": agent.epsilon_history,
        "q_values_history": agent.q_values_history,
        "final_eval": final_eval,
        "solved_episode": solved_episode,
        "best_reward": best_reward,
    }

    return agent, results


def plot_training_results(results):
    """
    Plot comprehensive training results

    Args:
        results: Dictionary containing training results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    rewards = results["rewards"]
    losses = results["losses"]
    epsilon_history = results["epsilon_history"]
    q_values_history = results["q_values_history"]

    ax = axes[0, 0]
    window = 20
    smoothed_rewards = pd.Series(rewards).rolling(window).mean()
    ax.plot(rewards, alpha=0.3, color="blue", label="Episode Rewards")
    ax.plot(
        smoothed_rewards, color="red", linewidth=2, label=f"Moving Average ({window})"
    )
    ax.set_title("Learning Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if losses:
        loss_window = 100
        smoothed_losses = pd.Series(losses).rolling(loss_window).mean()
        ax.plot(losses, alpha=0.3, color="orange", label="Training Loss")
        ax.plot(
            smoothed_losses,
            color="red",
            linewidth=2,
            label=f"Moving Average ({loss_window})",
        )
        ax.set_title("Training Loss")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epsilon_history)
    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if q_values_history:
        q_window = 100
        smoothed_q = pd.Series(q_values_history).rolling(q_window).mean()
        ax.plot(q_values_history, alpha=0.3, color="green", label="Q-Values")
        ax.plot(
            smoothed_q, color="red", linewidth=2, label=f"Moving Average ({q_window})"
        )
        ax.set_title("Q-Values Evolution")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Average Q-Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(rewards) >= 50:
        recent_rewards = rewards[-50:]
        ax.hist(recent_rewards, bins=20, alpha=0.7, edgecolor="black")
        ax.axvline(
            np.mean(recent_rewards),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(recent_rewards):.1f}",
        )
        ax.set_title("Reward Distribution (Last 50 Episodes)")
        ax.set_xlabel("Episode Reward")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    cumulative_reward = np.cumsum(rewards)
    ax.plot(cumulative_reward, linewidth=2)
    ax.set_title("Cumulative Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main experiment function"""
    agent, results = run_basic_dqn_experiment()

    print("Generating training plots...")
    plot_training_results(results)

    print("Performing Q-value analysis...")
    analyzer = PerformanceAnalyzer()
    analyzer.analyze_q_value_distributions(
        agent, gym.make("CartPole-v1"), num_samples=1000
    )

    print("\nExperiment completed successfully!")
    print("Results saved in the 'results' variable for further analysis.")

    return agent, results


if __name__ == "__main__":
    np.random.seed(42)

    agent, results = main()
