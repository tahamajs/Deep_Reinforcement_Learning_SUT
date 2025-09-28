"""
Double DQN Implementation
========================

This module implements Double DQN which addresses overestimation bias
in standard Q-learning by decoupling action selection from action evaluation.

Key Features:
- Double DQN agent with bias correction
- Overestimation bias analysis tools
- Performance comparison framework
- Synthetic environment for bias demonstration

Author: CA5 Implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dqn_base import DQNAgent, DQN, device
import random
import warnings

warnings.filterwarnings("ignore")


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent that addresses overestimation bias"""

    def __init__(self, state_size, action_size, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.agent_type = "Double DQN"

        self.q_value_estimates = []
        self.target_values = []

    def train_step(self):
        """Double DQN training step with bias correction"""
        if len(self.memory) < self.batch_size:
            return None

        experiences = self.memory.sample(self.batch_size)
        batch = self.experience_to_batch(experiences)

        states, actions, rewards, next_states, dones = batch

        current_q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.losses.append(loss.item())
        avg_q_value = current_q_values.mean().item()
        avg_target = target_q_values.mean().item()

        self.q_values.append(avg_q_value)
        self.q_value_estimates.append(avg_q_value)
        self.target_values.append(avg_target)

        return loss.item()


class OverestimationAnalysis:
    """Analyze and demonstrate overestimation bias in DQN vs Double DQN"""

    def __init__(self):
        self.results = {}

    def create_synthetic_environment(self, n_states=10, n_actions=5, noise_level=0.1):
        """Create synthetic environment to study overestimation"""
        true_q_values = np.random.uniform(0, 1, (n_states, n_actions))

        for s in range(n_states):
            best_action = np.argmax(true_q_values[s])
            true_q_values[s, best_action] += 0.2  # Boost best action

        return true_q_values, noise_level

    def simulate_estimation_bias(self, true_q_values, noise_level, n_estimates=1000):
        """Simulate Q-value estimation with noise"""
        n_states, n_actions = true_q_values.shape

        standard_estimates = []
        double_estimates = []

        for _ in range(n_estimates):
            noisy_q1 = true_q_values + np.random.normal(
                0, noise_level, true_q_values.shape
            )
            noisy_q2 = true_q_values + np.random.normal(
                0, noise_level, true_q_values.shape
            )

            standard_max = np.max(noisy_q1, axis=1)
            standard_estimates.append(standard_max)

            selected_actions = np.argmax(noisy_q1, axis=1)
            double_values = noisy_q2[np.arange(n_states), selected_actions]
            double_estimates.append(double_values)

        standard_estimates = np.array(standard_estimates)
        double_estimates = np.array(double_estimates)

        true_optimal = np.max(true_q_values, axis=1)

        return {
            "true_optimal": true_optimal,
            "standard_estimates": standard_estimates,
            "double_estimates": double_estimates,
            "standard_bias": np.mean(standard_estimates, axis=0) - true_optimal,
            "double_bias": np.mean(double_estimates, axis=0) - true_optimal,
        }

    def visualize_bias_analysis(self):
        """Visualize overestimation bias comparison"""
        true_q_values, noise_level = self.create_synthetic_environment()
        bias_results = self.simulate_estimation_bias(true_q_values, noise_level)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        states = range(len(bias_results["true_optimal"]))

        axes[0, 0].bar(
            [s - 0.2 for s in states],
            bias_results["standard_bias"],
            width=0.4,
            label="Standard DQN",
            alpha=0.7,
            color="red",
        )
        axes[0, 0].bar(
            [s + 0.2 for s in states],
            bias_results["double_bias"],
            width=0.4,
            label="Double DQN",
            alpha=0.7,
            color="blue",
        )
        axes[0, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[0, 0].set_title("Q-Value Estimation Bias by State")
        axes[0, 0].set_xlabel("State")
        axes[0, 0].set_ylabel("Bias (Estimated - True)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        state_0_standard = bias_results["standard_estimates"][:, 0]
        state_0_double = bias_results["double_estimates"][:, 0]
        true_value_0 = bias_results["true_optimal"][0]

        axes[0, 1].hist(
            state_0_standard,
            bins=30,
            alpha=0.6,
            label="Standard DQN",
            color="red",
            density=True,
        )
        axes[0, 1].hist(
            state_0_double,
            bins=30,
            alpha=0.6,
            label="Double DQN",
            color="blue",
            density=True,
        )
        axes[0, 1].axvline(
            true_value_0,
            color="black",
            linestyle="--",
            label=f"True Value: {true_value_0:.3f}",
        )
        axes[0, 1].set_title("Q-Value Estimate Distribution (State 0)")
        axes[0, 1].set_xlabel("Estimated Q-Value")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        noise_levels = np.linspace(0.01, 0.3, 20)
        standard_biases = []
        double_biases = []

        for noise in noise_levels:
            results = self.simulate_estimation_bias(true_q_values, noise, 200)
            standard_biases.append(np.mean(results["standard_bias"]))
            double_biases.append(np.mean(results["double_bias"]))

        axes[1, 0].plot(
            noise_levels,
            standard_biases,
            "o-",
            label="Standard DQN",
            color="red",
            linewidth=2,
        )
        axes[1, 0].plot(
            noise_levels,
            double_biases,
            "o-",
            label="Double DQN",
            color="blue",
            linewidth=2,
        )
        axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1, 0].set_title("Average Bias vs Noise Level")
        axes[1, 0].set_xlabel("Noise Level (σ)")
        axes[1, 0].set_ylabel("Average Bias")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        standard_vars = np.var(bias_results["standard_estimates"], axis=0)
        double_vars = np.var(bias_results["double_estimates"], axis=0)

        axes[1, 1].bar(
            [s - 0.2 for s in states],
            standard_vars,
            width=0.4,
            label="Standard DQN",
            alpha=0.7,
            color="red",
        )
        axes[1, 1].bar(
            [s + 0.2 for s in states],
            double_vars,
            width=0.4,
            label="Double DQN",
            alpha=0.7,
            color="blue",
        )
        axes[1, 1].set_title("Q-Value Estimate Variance by State")
        axes[1, 1].set_xlabel("State")
        axes[1, 1].set_ylabel("Variance")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("Overestimation Bias Analysis Summary:")
        print("=" * 50)
        print(
            f"Average Standard DQN Bias: {np.mean(bias_results['standard_bias']):.4f}"
        )
        print(f"Average Double DQN Bias: {np.mean(bias_results['double_bias']):.4f}")
        print(
            f"Bias Reduction: {(np.mean(bias_results['standard_bias']) - np.mean(bias_results['double_bias'])):.4f}"
        )
        print(f"Standard DQN Variance: {np.mean(standard_vars):.4f}")
        print(f"Double DQN Variance: {np.mean(double_vars):.4f}")

        return bias_results


class DQNComparison:
    """Compare Standard DQN vs Double DQN performance"""

    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

    def run_comparison(self, num_episodes=500, num_runs=3):
        """Run comparison between Standard DQN and Double DQN"""
        print("Starting DQN vs Double DQN Comparison...")
        print("=" * 60)

        standard_results = []
        double_results = []

        for run in range(num_runs):
            print(f"\\nRun {run + 1}/{num_runs}")

            print("Training Standard DQN...")
            standard_agent = DQNAgent(
                self.state_size, self.action_size, lr=0.0005, target_update_freq=1000
            )
            standard_scores, _ = standard_agent.train(
                self.env, num_episodes, print_every=num_episodes // 5
            )
            standard_results.append(standard_scores)

            print("Training Double DQN...")
            double_agent = DoubleDQNAgent(
                self.state_size, self.action_size, lr=0.0005, target_update_freq=1000
            )
            double_scores, _ = double_agent.train(
                self.env, num_episodes, print_every=num_episodes // 5
            )
            double_results.append(double_scores)

        return standard_results, double_results, standard_agent, double_agent

    def visualize_comparison(self, standard_results, double_results):
        """Visualize comparison results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        standard_mean = np.mean(standard_results, axis=0)
        double_mean = np.mean(double_results, axis=0)
        standard_std = np.std(standard_results, axis=0)
        double_std = np.std(double_results, axis=0)

        episodes = range(len(standard_mean))

        axes[0, 0].plot(
            episodes, standard_mean, color="red", label="Standard DQN", linewidth=2
        )
        axes[0, 0].fill_between(
            episodes,
            standard_mean - standard_std,
            standard_mean + standard_std,
            alpha=0.3,
            color="red",
        )

        axes[0, 0].plot(
            episodes, double_mean, color="blue", label="Double DQN", linewidth=2
        )
        axes[0, 0].fill_between(
            episodes,
            double_mean - double_std,
            double_mean + double_std,
            alpha=0.3,
            color="blue",
        )

        axes[0, 0].set_title("Learning Curves Comparison")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Episode Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        final_standard = [scores[-50:] for scores in standard_results]
        final_double = [scores[-50:] for scores in double_results]

        final_standard_flat = [score for run in final_standard for score in run]
        final_double_flat = [score for run in final_double for score in run]

        axes[0, 1].boxplot(
            [final_standard_flat, final_double_flat],
            labels=["Standard DQN", "Double DQN"],
        )
        axes[0, 1].set_title("Final Performance Distribution")
        axes[0, 1].set_ylabel("Episode Reward")
        axes[0, 1].grid(True, alpha=0.3)

        convergence_threshold = (
            np.mean(double_mean[-100:]) * 0.9
        )  # 90% of final performance

        standard_convergence = []
        double_convergence = []

        for standard_scores in standard_results:
            conv_episode = next(
                (
                    i
                    for i, score in enumerate(standard_scores)
                    if score >= convergence_threshold
                ),
                len(standard_scores),
            )
            standard_convergence.append(conv_episode)

        for double_scores in double_results:
            conv_episode = next(
                (
                    i
                    for i, score in enumerate(double_scores)
                    if score >= convergence_threshold
                ),
                len(double_scores),
            )
            double_convergence.append(conv_episode)

        axes[1, 0].bar(
            ["Standard DQN", "Double DQN"],
            [np.mean(standard_convergence), np.mean(double_convergence)],
            color=["red", "blue"],
            alpha=0.7,
        )
        axes[1, 0].set_title("Convergence Speed")
        axes[1, 0].set_ylabel("Episodes to Convergence")
        axes[1, 0].grid(True, alpha=0.3)

        improvement_window = 50
        standard_improvement = []
        double_improvement = []

        for i in range(improvement_window, len(episodes)):
            std_improvement = np.mean(
                standard_mean[i - improvement_window : i]
            ) - np.mean(standard_mean[:improvement_window])
            dbl_improvement = np.mean(
                double_mean[i - improvement_window : i]
            ) - np.mean(double_mean[:improvement_window])
            standard_improvement.append(std_improvement)
            double_improvement.append(dbl_improvement)

        imp_episodes = range(improvement_window, len(episodes))
        axes[1, 1].plot(
            imp_episodes,
            standard_improvement,
            color="red",
            label="Standard DQN",
            linewidth=2,
        )
        axes[1, 1].plot(
            imp_episodes,
            double_improvement,
            color="blue",
            label="Double DQN",
            linewidth=2,
        )
        axes[1, 1].set_title("Cumulative Improvement")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Improvement from Baseline")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Double DQN Implementation")
    print("=" * 40)

    agent = DoubleDQNAgent(state_size=4, action_size=2)
    print(f"Double DQN Agent created: {agent.agent_type}")

    print("\\nRunning Overestimation Bias Analysis...")
    bias_analysis = OverestimationAnalysis()
    bias_results = bias_analysis.visualize_bias_analysis()

    print("\\n✓ Double DQN implementation complete")
    print("✓ Overestimation bias analysis finished")
    print("✓ Comparison framework ready")
