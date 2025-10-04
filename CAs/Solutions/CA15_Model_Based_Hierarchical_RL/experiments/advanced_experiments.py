"""
Advanced Complex Experiments

This module contains advanced complex experiments for testing sophisticated RL algorithms:
- Multi-agent cooperation and competition experiments
- Hierarchical RL with curriculum learning
- Model-based RL with uncertainty quantification
- Continuous control with complex dynamics
- Real-world inspired robotics tasks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import random
import copy
from collections import deque, defaultdict
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import json
import os

# Import our advanced modules
from model_based_rl.advanced_algorithms import (
    ProbabilisticDynamicsModel,
    ModelBasedPolicyOptimization,
    DreamerAgent,
    ModelBasedMetaLearning,
    SafeModelBasedRL,
)
from hierarchical_rl.advanced_algorithms import (
    HIROAgent,
    AdvancedHAC,
    ContinuousOptionCritic,
    CurriculumScheduler,
    SubgoalGenerator,
)
from planning.advanced_algorithms import (
    AlphaZeroMCTS,
    ConstrainedMPC,
    VariationalLatentPlanner,
)
from environments.advanced_environments import (
    MultiAgentCooperationEnv,
    ContinuousControlEnv,
    HierarchicalTaskEnv,
    DynamicObstacleEnv,
)
from utils.advanced_visualization import AdvancedVisualizer, InteractiveVisualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdvancedExperimentRunner:
    """Advanced experiment runner for complex RL scenarios."""

    def __init__(self, results_dir="results", visualizations_dir="visualizations"):
        self.results_dir = results_dir
        self.visualizations_dir = visualizations_dir

        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)

        # Initialize visualizer
        self.visualizer = AdvancedVisualizer()
        self.interactive_visualizer = InteractiveVisualizer()

        # Experiment results storage
        self.experiment_results = {}

        # Performance metrics
        self.metrics = {
            "sample_efficiency": {},
            "final_performance": {},
            "computational_cost": {},
            "success_rate": {},
            "policy_entropy": {},
            "value_function": {},
        }

    def run_multi_agent_cooperation_experiment(self, num_agents=3, num_episodes=1000):
        """Run multi-agent cooperation experiment."""
        print("🤝 Running Multi-Agent Cooperation Experiment...")

        # Create environment
        env = MultiAgentCooperationEnv(num_agents=num_agents)

        # Initialize agents with different algorithms
        agents = {}
        agent_configs = {
            "HIRO": {
                "class": HIROAgent,
                "params": {
                    "state_dim": env.observation_space.shape[0],
                    "action_dim": env.action_space.n,
                    "subgoal_dim": 4,
                },
            },
            "HAC": {
                "class": AdvancedHAC,
                "params": {
                    "state_dim": env.observation_space.shape[0],
                    "action_dim": env.action_space.n,
                    "num_levels": 2,
                },
            },
            "OptionCritic": {
                "class": ContinuousOptionCritic,
                "params": {
                    "state_dim": env.observation_space.shape[0],
                    "action_dim": env.action_space.n,
                    "num_options": 4,
                },
            },
        }

        # Initialize agents
        for agent_name, config in agent_configs.items():
            agents[agent_name] = config["class"](**config["params"])

        # Training loop
        episode_rewards = defaultdict(list)
        episode_successes = defaultdict(list)
        episode_lengths = defaultdict(list)

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = np.zeros(num_agents)
            episode_length = 0
            done = False

            while not done:
                # Get actions from all agents
                actions = []
                for i, (agent_name, agent) in enumerate(agents.items()):
                    if agent_name == "HIRO":
                        subgoal = agent.get_subgoal(
                            torch.FloatTensor(state).unsqueeze(0)
                        )
                        action = agent.get_action(
                            torch.FloatTensor(state).unsqueeze(0), subgoal
                        )
                        actions.append(action.item())
                    elif agent_name == "HAC":
                        subgoals, actions_hac = agent.hierarchical_forward(
                            torch.FloatTensor(state).unsqueeze(0)
                        )
                        actions.append(actions_hac[0].item())
                    elif agent_name == "OptionCritic":
                        option, _ = agent.select_option(
                            torch.FloatTensor(state).unsqueeze(0)
                        )
                        action = agent.get_action(
                            torch.FloatTensor(state).unsqueeze(0), option
                        )
                        actions.append(action.item())

                # Take step
                next_state, rewards, dones, info = env.step(actions)

                # Update agents (simplified)
                for i, (agent_name, agent) in enumerate(agents.items()):
                    if hasattr(agent, "update"):
                        # Simplified update - in practice, you'd store experiences
                        pass

                episode_reward += rewards
                episode_length += 1
                state = next_state
                done = any(dones)

            # Store results
            for i, agent_name in enumerate(agents.keys()):
                episode_rewards[agent_name].append(episode_reward[i])
                episode_successes[agent_name].append(1 if episode_reward[i] > 5 else 0)
                episode_lengths[agent_name].append(episode_length)

            if episode % 100 == 0:
                avg_rewards = {
                    name: np.mean(rewards[-100:])
                    for name, rewards in episode_rewards.items()
                }
                print(f"Episode {episode}: Avg Rewards = {avg_rewards}")

        # Store results
        self.experiment_results["multi_agent_cooperation"] = {
            "episode_rewards": dict(episode_rewards),
            "episode_successes": dict(episode_successes),
            "episode_lengths": dict(episode_lengths),
        }

        # Create visualizations
        self._create_multi_agent_visualizations(episode_rewards, episode_successes)

        print("✅ Multi-Agent Cooperation Experiment Completed!")
        return self.experiment_results["multi_agent_cooperation"]

    def run_hierarchical_curriculum_experiment(self, num_levels=4, num_episodes=2000):
        """Run hierarchical RL with curriculum learning experiment."""
        print("📚 Running Hierarchical Curriculum Learning Experiment...")

        # Create hierarchical environment
        env = HierarchicalTaskEnv(num_levels=num_levels, tasks_per_level=3)

        # Initialize HAC agent with curriculum
        agent = AdvancedHAC(
            state_dim=env.state_dim, action_dim=env.action_dim, num_levels=num_levels
        )

        # Training data
        episode_rewards = []
        level_performances = defaultdict(list)
        curriculum_progress = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Get hierarchical action
                subgoals, actions = agent.hierarchical_forward(
                    torch.FloatTensor(state).unsqueeze(0)
                )

                # Take action
                action = actions[0].item()
                next_state, reward, done, info = env.step(action)

                episode_reward += reward
                state = next_state

                # Update curriculum
                if "task_completed" in info and info["task_completed"]:
                    agent.curriculum_scheduler.update_difficulty(
                        0.8
                    )  # High success rate

                # Store level performance
                current_level = info.get("current_level", 0)
                level_performances[current_level].append(reward)

            episode_rewards.append(episode_reward)
            curriculum_progress.append(agent.curriculum_scheduler.difficulty_level)

            if episode % 200 == 0:
                avg_reward = np.mean(episode_rewards[-200:])
                current_difficulty = agent.curriculum_scheduler.difficulty_level
                print(
                    f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Difficulty = {current_difficulty:.2f}"
                )

        # Store results
        self.experiment_results["hierarchical_curriculum"] = {
            "episode_rewards": episode_rewards,
            "level_performances": dict(level_performances),
            "curriculum_progress": curriculum_progress,
        }

        # Create visualizations
        self._create_hierarchical_visualizations(
            episode_rewards, level_performances, curriculum_progress
        )

        print("✅ Hierarchical Curriculum Learning Experiment Completed!")
        return self.experiment_results["hierarchical_curriculum"]

    def run_model_based_uncertainty_experiment(self, num_episodes=1500):
        """Run model-based RL with uncertainty quantification experiment."""
        print("🎯 Running Model-Based Uncertainty Quantification Experiment...")

        # Create continuous control environment
        env = ContinuousControlEnv(state_dim=6, action_dim=2)

        # Initialize different model-based agents
        agents = {
            "ProbabilisticModel": ProbabilisticDynamicsModel(
                state_dim=env.state_dim,
                action_dim=env.action_dim.shape[0],
                ensemble_size=5,
            ),
            "MBPO": ModelBasedPolicyOptimization(
                state_dim=env.state_dim, action_dim=env.action_dim.shape[0]
            ),
            "Dreamer": DreamerAgent(
                obs_dim=env.state_dim, action_dim=env.action_dim.shape[0]
            ),
            "SafeMBRL": SafeModelBasedRL(
                state_dim=env.state_dim,
                action_dim=env.action_dim.shape[0],
                constraint_dim=2,
            ),
        }

        # Training data
        agent_results = {}

        for agent_name, agent in agents.items():
            print(f"Training {agent_name}...")

            episode_rewards = []
            model_uncertainties = []
            prediction_errors = []

            for episode in range(num_episodes):
                state = env.reset()
                episode_reward = 0
                episode_uncertainties = []
                episode_errors = []

                for step in range(200):  # Max steps per episode
                    # Get action
                    if agent_name == "ProbabilisticModel":
                        # Use random action for dynamics model training
                        action = env.action_space.sample()
                    elif agent_name == "MBPO":
                        action = agent.get_action(torch.FloatTensor(state).unsqueeze(0))
                        action = action.squeeze().cpu().numpy()
                    elif agent_name == "Dreamer":
                        latent_mean, latent_std = agent.encode(
                            torch.FloatTensor(state).unsqueeze(0)
                        )
                        latent = agent.reparameterize(latent_mean, latent_std)
                        action = agent.get_action(latent)
                        action = action.squeeze().cpu().numpy()
                    elif agent_name == "SafeMBRL":
                        action = agent.get_safe_action(
                            torch.FloatTensor(state).unsqueeze(0)
                        )
                        action = action.squeeze().cpu().numpy()

                    # Take step
                    next_state, reward, done, info = env.step(action)

                    # Compute model uncertainty and prediction error
                    if agent_name == "ProbabilisticModel":
                        predictions = agent.forward(
                            torch.FloatTensor(state).unsqueeze(0),
                            torch.FloatTensor(action).unsqueeze(0),
                        )

                        # Compute uncertainty (variance across ensemble)
                        means = [pred[0] for pred in predictions]
                        mean_prediction = torch.mean(torch.stack(means), dim=0)
                        uncertainty = torch.var(torch.stack(means), dim=0).mean().item()

                        # Compute prediction error
                        true_next_state = torch.FloatTensor(next_state).unsqueeze(0)
                        prediction_error = F.mse_loss(
                            mean_prediction, true_next_state
                        ).item()

                        episode_uncertainties.append(uncertainty)
                        episode_errors.append(prediction_error)

                    episode_reward += reward
                    state = next_state

                    if done:
                        break

                episode_rewards.append(episode_reward)
                if episode_uncertainties:
                    model_uncertainties.append(np.mean(episode_uncertainties))
                if episode_errors:
                    prediction_errors.append(np.mean(episode_errors))

                if episode % 200 == 0:
                    avg_reward = np.mean(episode_rewards[-200:])
                    print(f"  Episode {episode}: Avg Reward = {avg_reward:.2f}")

            agent_results[agent_name] = {
                "rewards": episode_rewards,
                "uncertainties": model_uncertainties,
                "prediction_errors": prediction_errors,
            }

        # Store results
        self.experiment_results["model_based_uncertainty"] = agent_results

        # Create visualizations
        self._create_model_uncertainty_visualizations(agent_results)

        print("✅ Model-Based Uncertainty Quantification Experiment Completed!")
        return self.experiment_results["model_based_uncertainty"]

    def run_planning_comparison_experiment(self, num_episodes=800):
        """Run planning algorithms comparison experiment."""
        print("🧠 Running Planning Algorithms Comparison Experiment...")

        # Create environment
        env = DynamicObstacleEnv(grid_size=12, num_obstacles=4, num_goals=2)

        # Initialize planning agents
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Create dynamics model for planning
        dynamics_model = ProbabilisticDynamicsModel(state_dim, action_dim)

        agents = {
            "AlphaZeroMCTS": AlphaZeroMCTS(state_dim, action_dim, num_simulations=200),
            "ConstrainedMPC": ConstrainedMPC(dynamics_model, action_dim, horizon=8),
            "LatentPlanner": VariationalLatentPlanner(state_dim, action_dim),
        }

        # Training data
        agent_results = {}

        for agent_name, agent in agents.items():
            print(f"Training {agent_name}...")

            episode_rewards = []
            planning_times = []
            success_rates = []

            for episode in range(num_episodes):
                state = env.reset()
                episode_reward = 0
                episode_success = False
                start_time = time.time()

                for step in range(300):  # Max steps per episode
                    # Plan action
                    if agent_name == "AlphaZeroMCTS":
                        root = agent.search(state, dynamics_model)
                        action_probs = agent.get_action_probabilities(root)
                        action = np.argmax(action_probs)
                    elif agent_name == "ConstrainedMPC":
                        action = agent.plan(state)
                    elif agent_name == "LatentPlanner":
                        action = agent.plan_in_latent_space(
                            torch.FloatTensor(state).unsqueeze(0)
                        )
                        action = action.item()

                    # Take step
                    next_state, reward, done, info = env.step(action)

                    episode_reward += reward
                    state = next_state

                    if done:
                        episode_success = True
                        break

                planning_time = time.time() - start_time
                episode_rewards.append(episode_reward)
                planning_times.append(planning_time)
                success_rates.append(1 if episode_success else 0)

                if episode % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    avg_planning_time = np.mean(planning_times[-100:])
                    success_rate = np.mean(success_rates[-100:])
                    print(
                        f"  Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                        f"Planning Time = {avg_planning_time:.3f}s, Success Rate = {success_rate:.2f}"
                    )

            agent_results[agent_name] = {
                "rewards": episode_rewards,
                "planning_times": planning_times,
                "success_rates": success_rates,
            }

        # Store results
        self.experiment_results["planning_comparison"] = agent_results

        # Create visualizations
        self._create_planning_visualizations(agent_results)

        print("✅ Planning Algorithms Comparison Experiment Completed!")
        return self.experiment_results["planning_comparison"]

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark of all advanced algorithms."""
        print("🚀 Running Comprehensive Advanced RL Benchmark...")

        benchmark_results = {}

        # Run all experiments
        experiments = [
            ("multi_agent_cooperation", self.run_multi_agent_cooperation_experiment),
            ("hierarchical_curriculum", self.run_hierarchical_curriculum_experiment),
            ("model_based_uncertainty", self.run_model_based_uncertainty_experiment),
            ("planning_comparison", self.run_planning_comparison_experiment),
        ]

        for exp_name, exp_function in experiments:
            print(f"\n{'='*60}")
            print(f"Running {exp_name} experiment...")
            print(f"{'='*60}")

            try:
                results = exp_function()
                benchmark_results[exp_name] = results
                print(f"✅ {exp_name} completed successfully!")
            except Exception as e:
                print(f"❌ {exp_name} failed: {e}")
                benchmark_results[exp_name] = {"error": str(e)}

        # Create comprehensive dashboard
        self._create_comprehensive_dashboard(benchmark_results)

        # Save results
        self._save_benchmark_results(benchmark_results)

        print(f"\n🎉 Comprehensive Benchmark Completed!")
        print(f"📊 Results saved to: {self.results_dir}/")
        print(f"📈 Visualizations saved to: {self.visualizations_dir}/")

        return benchmark_results

    def _create_multi_agent_visualizations(self, episode_rewards, episode_successes):
        """Create multi-agent visualizations."""
        # Learning curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Multi-Agent Cooperation Results", fontsize=16)

        # Plot learning curves
        for agent_name, rewards in episode_rewards.items():
            episodes = np.arange(len(rewards))
            axes[0, 0].plot(episodes, rewards, label=agent_name, alpha=0.7)

        axes[0, 0].set_title("Learning Curves")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot success rates
        for agent_name, successes in episode_successes.items():
            episodes = np.arange(len(successes))
            success_rate = np.cumsum(successes) / (episodes + 1)
            axes[0, 1].plot(episodes, success_rate, label=agent_name, alpha=0.7)

        axes[0, 1].set_title("Success Rate")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Success Rate")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Final performance comparison
        final_performances = [
            np.mean(rewards[-100:]) for rewards in episode_rewards.values()
        ]
        agent_names = list(episode_rewards.keys())

        axes[1, 0].bar(agent_names, final_performances, alpha=0.7)
        axes[1, 0].set_title("Final Performance")
        axes[1, 0].set_ylabel("Average Reward (Last 100 Episodes)")
        axes[1, 0].grid(True, alpha=0.3)

        # Sample efficiency
        sample_efficiencies = []
        for rewards in episode_rewards.values():
            threshold = np.mean(rewards) * 0.8
            episodes_to_threshold = np.where(np.array(rewards) >= threshold)[0]
            if len(episodes_to_threshold) > 0:
                sample_efficiencies.append(episodes_to_threshold[0])
            else:
                sample_efficiencies.append(len(rewards))

        axes[1, 1].bar(agent_names, sample_efficiencies, alpha=0.7)
        axes[1, 1].set_title("Sample Efficiency")
        axes[1, 1].set_ylabel("Episodes to 80% Performance")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.visualizations_dir}/multi_agent_cooperation_results.png", dpi=300
        )
        plt.show()

    def _create_hierarchical_visualizations(
        self, episode_rewards, level_performances, curriculum_progress
    ):
        """Create hierarchical visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Hierarchical Curriculum Learning Results", fontsize=16)

        # Learning curve
        episodes = np.arange(len(episode_rewards))
        axes[0, 0].plot(episodes, episode_rewards, alpha=0.7)
        axes[0, 0].set_title("Overall Learning Curve")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)

        # Level-wise performance
        for level, performances in level_performances.items():
            if performances:
                axes[0, 1].plot(performances, label=f"Level {level}", alpha=0.7)

        axes[0, 1].set_title("Performance by Level")
        axes[0, 1].set_xlabel("Task")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Curriculum progress
        axes[1, 0].plot(episodes, curriculum_progress, alpha=0.7, color="red")
        axes[1, 0].set_title("Curriculum Difficulty Progress")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Difficulty Level")
        axes[1, 0].grid(True, alpha=0.3)

        # Performance vs curriculum
        axes[1, 1].scatter(curriculum_progress, episode_rewards, alpha=0.5)
        axes[1, 1].set_title("Performance vs Curriculum Difficulty")
        axes[1, 1].set_xlabel("Curriculum Difficulty")
        axes[1, 1].set_ylabel("Episode Reward")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.visualizations_dir}/hierarchical_curriculum_results.png", dpi=300
        )
        plt.show()

    def _create_model_uncertainty_visualizations(self, agent_results):
        """Create model uncertainty visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Model-Based Uncertainty Quantification Results", fontsize=16)

        # Learning curves
        for agent_name, results in agent_results.items():
            episodes = np.arange(len(results["rewards"]))
            axes[0, 0].plot(episodes, results["rewards"], label=agent_name, alpha=0.7)

        axes[0, 0].set_title("Learning Curves")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Model uncertainties
        if "ProbabilisticModel" in agent_results:
            uncertainties = agent_results["ProbabilisticModel"]["uncertainties"]
            if uncertainties:
                episodes = np.arange(len(uncertainties))
                axes[0, 1].plot(episodes, uncertainties, alpha=0.7, color="red")
                axes[0, 1].set_title("Model Uncertainty Over Time")
                axes[0, 1].set_xlabel("Episode")
                axes[0, 1].set_ylabel("Uncertainty")
                axes[0, 1].grid(True, alpha=0.3)

        # Prediction errors
        if "ProbabilisticModel" in agent_results:
            errors = agent_results["ProbabilisticModel"]["prediction_errors"]
            if errors:
                episodes = np.arange(len(errors))
                axes[1, 0].plot(episodes, errors, alpha=0.7, color="blue")
                axes[1, 0].set_title("Prediction Error Over Time")
                axes[1, 0].set_xlabel("Episode")
                axes[1, 0].set_ylabel("Prediction Error")
                axes[1, 0].grid(True, alpha=0.3)

        # Final performance comparison
        final_performances = []
        agent_names = []
        for agent_name, results in agent_results.items():
            if results["rewards"]:
                final_performances.append(np.mean(results["rewards"][-100:]))
                agent_names.append(agent_name)

        if final_performances:
            axes[1, 1].bar(agent_names, final_performances, alpha=0.7)
            axes[1, 1].set_title("Final Performance Comparison")
            axes[1, 1].set_ylabel("Average Reward (Last 100 Episodes)")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.visualizations_dir}/model_uncertainty_results.png", dpi=300)
        plt.show()

    def _create_planning_visualizations(self, agent_results):
        """Create planning visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Planning Algorithms Comparison Results", fontsize=16)

        # Learning curves
        for agent_name, results in agent_results.items():
            episodes = np.arange(len(results["rewards"]))
            axes[0, 0].plot(episodes, results["rewards"], label=agent_name, alpha=0.7)

        axes[0, 0].set_title("Learning Curves")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Planning times
        for agent_name, results in agent_results.items():
            episodes = np.arange(len(results["planning_times"]))
            axes[0, 1].plot(
                episodes, results["planning_times"], label=agent_name, alpha=0.7
            )

        axes[0, 1].set_title("Planning Time per Episode")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Planning Time (seconds)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Success rates
        for agent_name, results in agent_results.items():
            episodes = np.arange(len(results["success_rates"]))
            success_rate = np.cumsum(results["success_rates"]) / (episodes + 1)
            axes[1, 0].plot(episodes, success_rate, label=agent_name, alpha=0.7)

        axes[1, 0].set_title("Success Rate Over Time")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Success Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Performance vs computational cost
        final_performances = []
        avg_planning_times = []
        agent_names = []

        for agent_name, results in agent_results.items():
            if results["rewards"] and results["planning_times"]:
                final_performances.append(np.mean(results["rewards"][-100:]))
                avg_planning_times.append(np.mean(results["planning_times"][-100:]))
                agent_names.append(agent_name)

        if final_performances:
            scatter = axes[1, 1].scatter(
                avg_planning_times, final_performances, s=100, alpha=0.7
            )
            for i, name in enumerate(agent_names):
                axes[1, 1].annotate(
                    name,
                    (avg_planning_times[i], final_performances[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            axes[1, 1].set_title("Performance vs Computational Cost")
            axes[1, 1].set_xlabel("Average Planning Time (seconds)")
            axes[1, 1].set_ylabel("Final Performance")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.visualizations_dir}/planning_comparison_results.png", dpi=300
        )
        plt.show()

    def _create_comprehensive_dashboard(self, benchmark_results):
        """Create comprehensive dashboard."""
        # Create summary statistics
        summary_stats = {}

        for exp_name, results in benchmark_results.items():
            if "error" not in results:
                summary_stats[exp_name] = {
                    "status": "completed",
                    "num_episodes": len(results.get("episode_rewards", [])),
                    "final_performance": (
                        np.mean(results.get("episode_rewards", [])[-100:])
                        if results.get("episode_rewards")
                        else 0
                    ),
                }
            else:
                summary_stats[exp_name] = {
                    "status": "failed",
                    "error": results["error"],
                }

        # Create dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Comprehensive Advanced RL Benchmark Dashboard", fontsize=16)

        # Experiment status
        completed_exps = [
            name
            for name, stats in summary_stats.items()
            if stats["status"] == "completed"
        ]
        failed_exps = [
            name for name, stats in summary_stats.items() if stats["status"] == "failed"
        ]

        axes[0, 0].pie(
            [len(completed_exps), len(failed_exps)],
            labels=["Completed", "Failed"],
            autopct="%1.1f%%",
            colors=["lightgreen", "lightcoral"],
        )
        axes[0, 0].set_title("Experiment Status")

        # Final performance comparison
        exp_names = []
        performances = []
        for exp_name, stats in summary_stats.items():
            if stats["status"] == "completed":
                exp_names.append(exp_name)
                performances.append(stats["final_performance"])

        if performances:
            axes[0, 1].bar(exp_names, performances, alpha=0.7)
            axes[0, 1].set_title("Final Performance by Experiment")
            axes[0, 1].set_ylabel("Average Reward")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # Episode counts
        episode_counts = []
        for exp_name, stats in summary_stats.items():
            if stats["status"] == "completed":
                episode_counts.append(stats["num_episodes"])

        if episode_counts:
            axes[1, 0].bar(exp_names, episode_counts, alpha=0.7, color="orange")
            axes[1, 0].set_title("Training Episodes by Experiment")
            axes[1, 0].set_ylabel("Number of Episodes")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # Summary text
        summary_text = f"""
        Comprehensive Benchmark Summary:
        
        Total Experiments: {len(benchmark_results)}
        Completed: {len(completed_exps)}
        Failed: {len(failed_exps)}
        
        Completed Experiments:
        {chr(10).join([f"• {exp}" for exp in completed_exps])}
        
        Failed Experiments:
        {chr(10).join([f"• {exp}" for exp in failed_exps])}
        """

        axes[1, 1].text(
            0.1,
            0.9,
            summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        axes[1, 1].set_title("Summary")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(f"{self.visualizations_dir}/comprehensive_dashboard.png", dpi=300)
        plt.show()

    def _save_benchmark_results(self, benchmark_results):
        """Save benchmark results to files."""
        # Save JSON results
        with open(f"{self.results_dir}/benchmark_results.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for exp_name, results in benchmark_results.items():
                json_results[exp_name] = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        json_results[exp_name][key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, np.ndarray):
                                json_results[exp_name][key][
                                    sub_key
                                ] = sub_value.tolist()
                            else:
                                json_results[exp_name][key][sub_key] = sub_value
                    elif isinstance(value, np.ndarray):
                        json_results[exp_name][key] = value.tolist()
                    else:
                        json_results[exp_name][key] = value

            json.dump(json_results, f, indent=2)

        # Save summary report
        with open(f"{self.results_dir}/benchmark_summary.md", "w") as f:
            f.write("# Advanced RL Benchmark Summary\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for exp_name, results in benchmark_results.items():
                f.write(f"## {exp_name.replace('_', ' ').title()}\n\n")

                if "error" in results:
                    f.write(f"**Status:** Failed\n")
                    f.write(f"**Error:** {results['error']}\n\n")
                else:
                    f.write(f"**Status:** Completed\n")

                    if "episode_rewards" in results:
                        rewards = results["episode_rewards"]
                        if isinstance(rewards, dict):
                            f.write(f"**Agents:** {list(rewards.keys())}\n")
                            for agent_name, agent_rewards in rewards.items():
                                f.write(
                                    f"- {agent_name}: {len(agent_rewards)} episodes, "
                                    f"final performance: {np.mean(agent_rewards[-100:]):.2f}\n"
                                )
                        else:
                            f.write(f"**Episodes:** {len(rewards)}\n")
                            f.write(
                                f"**Final Performance:** {np.mean(rewards[-100:]):.2f}\n"
                            )

                    f.write("\n")

        print(f"📊 Results saved to {self.results_dir}/benchmark_results.json")
        print(f"📋 Summary saved to {self.results_dir}/benchmark_summary.md")


def main():
    """Main function to run advanced experiments."""
    print("🚀 Starting Advanced RL Experiments...")

    # Initialize experiment runner
    runner = AdvancedExperimentRunner()

    # Run comprehensive benchmark
    results = runner.run_comprehensive_benchmark()

    print("\n🎉 All Advanced RL Experiments Completed!")
    print("📊 Check the 'results/' and 'visualizations/' directories for outputs.")


if __name__ == "__main__":
    main()
