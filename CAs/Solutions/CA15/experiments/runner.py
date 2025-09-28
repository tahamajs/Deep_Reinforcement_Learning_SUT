"""
Reinforcement Learning Experiments

This module contains experiment runners and evaluation frameworks for comparing
different RL algorithms and analyzing their performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExperimentRunner:
    """Unified experiment runner for all algorithms."""

    def __init__(self, env_class, env_kwargs=None):
        self.env_class = env_class
        self.env_kwargs = env_kwargs or {}
        self.results = {}

    def run_experiment(self, agent_configs, num_episodes=500, num_seeds=3):
        """Run experiment with multiple agents and seeds."""
        results = {}

        for agent_name, agent_config in agent_configs.items():
            print(f"\n🔄 Running experiment for {agent_name}...")
            agent_results = []

            for seed in range(num_seeds):
                print(f"  Seed {seed + 1}/{num_seeds}")

                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)

                env = self.env_class(**self.env_kwargs)
                agent = agent_config["class"](**agent_config["params"])

                episode_rewards = []
                episode_lengths = []
                model_losses = []
                planning_times = []

                for episode in range(num_episodes):
                    state = env.reset()
                    episode_reward = 0
                    episode_length = 0
                    done = False

                    start_time = time.time()

                    while not done:
                        if hasattr(agent, "get_action"):
                            action = agent.get_action(state)
                        elif hasattr(agent, "plan_action"):
                            action = agent.plan_action(state)
                        else:
                            action = np.random.randint(
                                env.action_space.n
                                if hasattr(env, "action_space")
                                else 4
                            )

                        if hasattr(env, "step"):
                            next_state, reward, done, info = env.step(action)
                        else:
                            next_state, reward, done = (
                                state,
                                np.random.randn(),
                                np.random.random() < 0.1,
                            )
                            info = {}

                        episode_reward += reward
                        episode_length += 1

                        if hasattr(agent, "store_experience"):
                            agent.store_experience(
                                state, action, reward, next_state, done
                            )

                        if hasattr(agent, "update_q_function"):
                            q_loss = agent.update_q_function()
                        elif hasattr(agent, "train_step"):
                            losses = agent.train_step()

                        if hasattr(agent, "update_model"):
                            model_loss = agent.update_model()
                            model_losses.append(model_loss)

                        if hasattr(agent, "planning_step"):
                            agent.planning_step()

                        state = next_state

                        if episode_length > 500:  # Timeout
                            break

                    planning_time = time.time() - start_time
                    planning_times.append(planning_time)

                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)

                    if (episode + 1) % 100 == 0:
                        avg_reward = np.mean(episode_rewards[-100:])
                        print(
                            f"    Episode {episode + 1}: Avg Reward = {avg_reward:.2f}"
                        )

                agent_results.append(
                    {
                        "rewards": episode_rewards,
                        "lengths": episode_lengths,
                        "model_losses": model_losses,
                        "planning_times": planning_times,
                        "final_performance": np.mean(episode_rewards[-50:]),
                    }
                )

            results[agent_name] = agent_results

        self.results = results
        return results

    def analyze_results(self):
        """Analyze and visualize experiment results."""
        if not self.results:
            print("❌ No results to analyze. Run experiment first.")
            return

        print("\n📊 Experiment Results Analysis")
        print("=" * 50)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model-Based vs Model-Free Comparison", fontsize=16)

        ax1 = axes[0, 0]
        for agent_name, agent_results in self.results.items():
            all_rewards = [result["rewards"] for result in agent_results]
            min_length = min(len(rewards) for rewards in all_rewards)

            rewards_array = np.array([rewards[:min_length] for rewards in all_rewards])
            mean_rewards = np.mean(rewards_array, axis=0)
            std_rewards = np.std(rewards_array, axis=0)

            episodes = np.arange(min_length)
            ax1.plot(episodes, mean_rewards, label=agent_name, linewidth=2)
            ax1.fill_between(
                episodes,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.3,
            )

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Average Reward")
        ax1.set_title("Learning Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        threshold = -100  # Adjust based on environment

        agent_names = []
        sample_efficiencies = []
        sample_stds = []

        for agent_name, agent_results in self.results.items():
            episodes_to_threshold = []

            for result in agent_results:
                rewards = result["rewards"]
                moving_avg = np.convolve(rewards, np.ones(50) / 50, mode="valid")
                threshold_idx = np.where(moving_avg >= threshold)[0]

                if len(threshold_idx) > 0:
                    episodes_to_threshold.append(threshold_idx[0] + 50)
                else:
                    episodes_to_threshold.append(len(rewards))  # Didn't reach threshold

            agent_names.append(agent_name)
            sample_efficiencies.append(np.mean(episodes_to_threshold))
            sample_stds.append(np.std(episodes_to_threshold))

        bars = ax2.bar(
            agent_names,
            sample_efficiencies,
            yerr=sample_stds,
            capsize=5,
            color=["skyblue", "lightcoral", "lightgreen", "gold"][: len(agent_names)],
        )
        ax2.set_ylabel("Episodes to Threshold")
        ax2.set_title("Sample Efficiency")
        ax2.tick_params(axis="x", rotation=45)

        ax3 = axes[1, 0]

        final_performances = []
        final_stds = []

        for agent_name, agent_results in self.results.items():
            performances = [result["final_performance"] for result in agent_results]
            final_performances.append(np.mean(performances))
            final_stds.append(np.std(performances))

        bars = ax3.bar(
            agent_names,
            final_performances,
            yerr=final_stds,
            capsize=5,
            color=["skyblue", "lightcoral", "lightgreen", "gold"][: len(agent_names)],
        )
        ax3.set_ylabel("Final Average Reward")
        ax3.set_title("Final Performance")
        ax3.tick_params(axis="x", rotation=45)

        ax4 = axes[1, 1]

        planning_times = []
        time_stds = []

        for agent_name, agent_results in self.results.items():
            times = []
            for result in agent_results:
                if result["planning_times"]:
                    times.extend(result["planning_times"])

            if times:
                planning_times.append(np.mean(times))
                time_stds.append(np.std(times))
            else:
                planning_times.append(0)
                time_stds.append(0)

        bars = ax4.bar(
            agent_names,
            planning_times,
            yerr=time_stds,
            capsize=5,
            color=["skyblue", "lightcoral", "lightgreen", "gold"][: len(agent_names)],
        )
        ax4.set_ylabel("Average Planning Time (s)")
        ax4.set_title("Computational Overhead")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

        print("\n📈 Summary Statistics:")
        for agent_name, agent_results in self.results.items():
            performances = [result["final_performance"] for result in agent_results]
            mean_perf = np.mean(performances)
            std_perf = np.std(performances)

            print(f"\n{agent_name}:")
            print(f"  Final Performance: {mean_perf:.2f} ± {std_perf:.2f}")

            episodes_to_threshold = []
            for result in agent_results:
                rewards = result["rewards"]
                moving_avg = np.convolve(rewards, np.ones(50) / 50, mode="valid")
                threshold_idx = np.where(moving_avg >= threshold)[0]
                if len(threshold_idx) > 0:
                    episodes_to_threshold.append(threshold_idx[0] + 50)

            if episodes_to_threshold:
                mean_efficiency = np.mean(episodes_to_threshold)
                std_efficiency = np.std(episodes_to_threshold)
                print(
                    f"  Sample Efficiency: {mean_efficiency:.0f} ± {std_efficiency:.0f} episodes"
                )
