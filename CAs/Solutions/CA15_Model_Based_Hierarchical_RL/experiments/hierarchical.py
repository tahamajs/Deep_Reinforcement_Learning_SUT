"""
Hierarchical Reinforcement Learning Experiments

This module contains specialized experiments for testing hierarchical RL algorithms,
including goal-conditioned learning and multi-goal navigation tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HierarchicalRLExperiment:
    """Experiment to demonstrate hierarchical RL benefits."""

    def __init__(self):
        self.results = {}

    def create_multi_goal_environment(self, size=12, num_goals=4):
        """Create a complex multi-goal environment."""
        from ..hierarchical_rl.environments import HierarchicalRLEnvironment

        return HierarchicalRLEnvironment(size=size, num_goals=num_goals)

    def run_hierarchical_experiment(self, num_episodes=300, num_seeds=3):
        """Run hierarchical RL experiment with multiple approaches."""

        print("🏗️ Running Hierarchical RL Experiment...")
        print("🎯 Testing: Goal-Conditioned RL vs Standard RL vs Hierarchical AC")

        env_size = 10
        num_goals = 3

        from ..hierarchical_rl.algorithms import GoalConditionedAgent
        from ..model_based_rl.algorithms import DynaQAgent

        agent_configs = {
            "Goal-Conditioned Agent": {
                "class": GoalConditionedAgent,
                "params": {
                    "state_dim": env_size * env_size,
                    "action_dim": 4,
                    "goal_dim": env_size * env_size,
                },
            },
            "Standard DQN-like": {
                "class": DynaQAgent,
                "params": {
                    "state_dim": env_size * env_size,
                    "action_dim": 4,
                    "lr": 1e-3,
                },
            },
        }

        results = {}

        for agent_name, agent_config in agent_configs.items():
            print(f"\n🔄 Testing {agent_name}...")
            agent_results = []

            for seed in range(num_seeds):
                print(f"  Seed {seed + 1}/{num_seeds}")

                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)

                env = self.create_multi_goal_environment(env_size, num_goals)
                agent = agent_config["class"](**agent_config["params"])

                episode_rewards = []
                goal_achievements = []
                episode_lengths = []
                skill_reuse_success = []

                for episode in range(num_episodes):
                    state = env.reset()
                    episode_reward = 0
                    episode_length = 0
                    goals_reached = 0
                    done = False

                    if agent_name == "Goal-Conditioned Agent":
                        episode_states = [state]
                        episode_actions = []
                        episode_goals = []

                        current_goal = np.zeros_like(state)
                        if hasattr(env, "goals") and len(env.goals) > 0:
                            goal_pos = env.goals[env.current_goal_idx]
                            goal_idx = goal_pos[0] * env_size + goal_pos[1]
                            current_goal[goal_idx] = 1.0

                    while not done and episode_length < 200:
                        if agent_name == "Goal-Conditioned Agent":
                            action = agent.get_action(state, current_goal)
                            episode_goals.append(current_goal.copy())
                        else:
                            action = agent.get_action(state)

                        next_state, reward, done, info = env.step(action)
                        episode_reward += reward
                        episode_length += 1

                        if "goals_completed" in info:
                            goals_reached = info["goals_completed"]

                        if agent_name == "Goal-Conditioned Agent":
                            episode_states.append(next_state)
                            episode_actions.append(action)
                        else:
                            if hasattr(agent, "store_experience"):
                                agent.store_experience(
                                    state, action, reward, next_state, done
                                )
                            if hasattr(agent, "update_q_function"):
                                agent.update_q_function()
                            if hasattr(agent, "update_model"):
                                agent.update_model()

                        state = next_state

                        if agent_name == "Goal-Conditioned Agent" and hasattr(
                            env, "goals"
                        ):
                            if env.current_goal_idx < len(env.goals):
                                goal_pos = env.goals[env.current_goal_idx]
                                current_goal = np.zeros_like(state)
                                goal_idx = goal_pos[0] * env_size + goal_pos[1]
                                current_goal[goal_idx] = 1.0

                    if (
                        agent_name == "Goal-Conditioned Agent"
                        and len(episode_states) > 1
                    ):
                        final_achieved_goal = episode_states[-1]
                        agent.store_episode(
                            episode_states,
                            episode_actions,
                            episode_goals,
                            final_achieved_goal,
                        )

                        for _ in range(10):
                            agent.train_step(batch_size=32)

                    episode_rewards.append(episode_reward)
                    goal_achievements.append(goals_reached / num_goals)
                    episode_lengths.append(episode_length)

                    if episode % 50 == 0 and episode > 0:
                        skill_reuse_score = self._test_skill_reuse(
                            agent, env, agent_name
                        )
                        skill_reuse_success.append(skill_reuse_score)

                    if (episode + 1) % 100 == 0:
                        avg_reward = np.mean(episode_rewards[-50:])
                        avg_goals = np.mean(goal_achievements[-50:])
                        print(
                            f"    Episode {episode + 1}: Reward={avg_reward:.2f}, Goals={avg_goals:.2f}"
                        )

                agent_results.append(
                    {
                        "rewards": episode_rewards,
                        "goal_achievements": goal_achievements,
                        "lengths": episode_lengths,
                        "skill_reuse": skill_reuse_success,
                        "final_performance": np.mean(episode_rewards[-30:]),
                        "final_goal_rate": np.mean(goal_achievements[-30:]),
                    }
                )

            results[agent_name] = agent_results

        self.results = results
        return results

    def _test_skill_reuse(self, agent, env, agent_name):
        """Test how well agent transfers skills to new goal configurations."""
        test_env = self.create_multi_goal_environment(env.size, env.num_goals)

        success_count = 0
        test_episodes = 5

        for _ in range(test_episodes):
            state = test_env.reset()
            done = False
            steps = 0
            goals_reached = 0

            if agent_name == "Goal-Conditioned Agent":
                current_goal = np.zeros_like(state)
                if hasattr(test_env, "goals") and len(test_env.goals) > 0:
                    goal_pos = test_env.goals[0]
                    goal_idx = goal_pos[0] * test_env.size + goal_pos[1]
                    current_goal[goal_idx] = 1.0

            while not done and steps < 100:
                if agent_name == "Goal-Conditioned Agent":
                    action = agent.get_action(state, current_goal, deterministic=True)
                else:
                    action = agent.get_action(state, epsilon=0.1)  # Slight exploration

                next_state, reward, done, info = test_env.step(action)
                steps += 1

                if "goals_completed" in info:
                    goals_reached = info["goals_completed"]

                state = next_state

            if goals_reached > 0 and steps < 80:
                success_count += 1

        return success_count / test_episodes

    def visualize_hierarchical_results(self):
        """Visualize hierarchical RL experiment results."""
        if not self.results:
            print("❌ No results to visualize. Run experiment first.")
            return

        print("\n📊 Hierarchical RL Results Analysis")
        print("=" * 50)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Hierarchical RL Performance Analysis", fontsize=16)

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
        for agent_name, agent_results in self.results.items():
            all_goals = [result["goal_achievements"] for result in agent_results]
            min_length = min(len(goals) for goals in all_goals)

            goals_array = np.array([goals[:min_length] for goals in all_goals])
            mean_goals = np.mean(goals_array, axis=0)
            std_goals = np.std(goals_array, axis=0)

            episodes = np.arange(min_length)
            ax2.plot(episodes, mean_goals, label=agent_name, linewidth=2)
            ax2.fill_between(
                episodes, mean_goals - std_goals, mean_goals + std_goals, alpha=0.3
            )

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Goal Achievement Rate")
        ax2.set_title("Goal Completion Progress")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[0, 2]
        agent_names = list(self.results.keys())
        skill_reuse_means = []
        skill_reuse_stds = []

        for agent_name, agent_results in self.results.items():
            all_reuse = []
            for result in agent_results:
                if result["skill_reuse"]:
                    all_reuse.extend(result["skill_reuse"])

            if all_reuse:
                skill_reuse_means.append(np.mean(all_reuse))
                skill_reuse_stds.append(np.std(all_reuse))
            else:
                skill_reuse_means.append(0)
                skill_reuse_stds.append(0)

        bars = ax3.bar(
            agent_names,
            skill_reuse_means,
            yerr=skill_reuse_stds,
            capsize=5,
            color=["lightblue", "lightcoral"],
        )
        ax3.set_ylabel("Skill Transfer Success Rate")
        ax3.set_title("Skill Reuse Capability")
        ax3.tick_params(axis="x", rotation=45)

        ax4 = axes[1, 0]
        length_means = []
        length_stds = []

        for agent_name, agent_results in self.results.items():
            all_lengths = []
            for result in agent_results:
                all_lengths.extend(result["lengths"][-50:])  # Last 50 episodes

            length_means.append(np.mean(all_lengths))
            length_stds.append(np.std(all_lengths))

        bars = ax4.bar(
            agent_names,
            length_means,
            yerr=length_stds,
            capsize=5,
            color=["lightblue", "lightcoral"],
        )
        ax4.set_ylabel("Average Episode Length")
        ax4.set_title("Efficiency (Lower is Better)")
        ax4.tick_params(axis="x", rotation=45)

        ax5 = axes[1, 1]
        final_rewards = []
        final_stds = []

        for agent_name, agent_results in self.results.items():
            performances = [result["final_performance"] for result in agent_results]
            final_rewards.append(np.mean(performances))
            final_stds.append(np.std(performances))

        bars = ax5.bar(
            agent_names,
            final_rewards,
            yerr=final_stds,
            capsize=5,
            color=["lightblue", "lightcoral"],
        )
        ax5.set_ylabel("Final Average Reward")
        ax5.set_title("Final Performance")
        ax5.tick_params(axis="x", rotation=45)

        ax6 = axes[1, 2]
        final_goal_rates = []
        goal_rate_stds = []

        for agent_name, agent_results in self.results.items():
            goal_rates = [result["final_goal_rate"] for result in agent_results]
            final_goal_rates.append(np.mean(goal_rates))
            goal_rate_stds.append(np.std(goal_rates))

        bars = ax6.bar(
            agent_names,
            final_goal_rates,
            yerr=goal_rate_stds,
            capsize=5,
            color=["lightblue", "lightcoral"],
        )
        ax6.set_ylabel("Final Goal Achievement Rate")
        ax6.set_title("Multi-Goal Success Rate")
        ax6.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

        print("\n📈 Hierarchical RL Analysis Summary:")
        for agent_name, agent_results in self.results.items():
            final_rewards = [result["final_performance"] for result in agent_results]
            final_goals = [result["final_goal_rate"] for result in agent_results]

            print(f"\n{agent_name}:")
            print(
                f"  Final Reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}"
            )
            print(
                f"  Goal Success Rate: {np.mean(final_goals):.3f} ± {np.std(final_goals):.3f}"
            )
            print(f"  Skill Transfer: {np.mean(skill_reuse_means):.3f}")
