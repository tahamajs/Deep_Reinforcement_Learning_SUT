"""
Complex Experiments and Deep Comparisons for Advanced DQN Methods
Includes: Multi-environment testing, ablation studies, and comprehensive evaluations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
import json
import os
import time
from datetime import datetime
import gym
from collections import defaultdict
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


class ComplexExperimentRunner:
    """Advanced experiment runner with comprehensive analysis"""

    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.results = {}
        self.experiment_history = []

    def run_ablation_study(
        self,
        base_agent_class,
        ablation_configs: Dict[str, Dict],
        env_name: str = "CartPole-v1",
        num_episodes: int = 1000,
        num_runs: int = 5,
    ) -> Dict[str, Any]:
        """Run ablation study to understand component contributions"""

        print(f"Starting ablation study with {len(ablation_configs)} configurations...")

        ablation_results = {}

        for config_name, config in ablation_configs.items():
            print(f"\nTesting configuration: {config_name}")

            run_results = []

            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}")

                # Create environment
                env = gym.make(env_name)

                # Create agent with configuration
                agent = base_agent_class(
                    state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.n,
                    **config,
                )

                # Train agent
                training_results = self._train_agent_detailed(agent, env, num_episodes)

                # Evaluate agent
                eval_results = self._evaluate_agent_detailed(
                    agent, env, num_episodes=100
                )

                run_results.append(
                    {
                        "training": training_results,
                        "evaluation": eval_results,
                        "run_id": run,
                    }
                )

                env.close()

            # Aggregate results across runs
            ablation_results[config_name] = self._aggregate_results(run_results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.base_dir, f"ablation_study_{timestamp}.json")

        with open(results_file, "w") as f:
            json.dump(ablation_results, f, indent=2)

        # Generate ablation analysis
        self._generate_ablation_analysis(ablation_results, timestamp)

        return ablation_results

    def run_multi_environment_study(
        self,
        agent_classes: Dict[str, Any],
        environments: List[str],
        num_episodes: int = 1000,
    ) -> Dict[str, Any]:
        """Run comprehensive study across multiple environments"""

        print(
            f"Starting multi-environment study with {len(agent_classes)} agents and {len(environments)} environments..."
        )

        multi_env_results = {}

        for env_name in environments:
            print(f"\nTesting environment: {env_name}")
            multi_env_results[env_name] = {}

            for agent_name, agent_class in agent_classes.items():
                print(f"  Testing agent: {agent_name}")

                try:
                    # Create environment
                    env = gym.make(env_name)

                    # Create agent
                    agent = agent_class(
                        state_dim=env.observation_space.shape[0],
                        action_dim=env.action_space.n,
                    )

                    # Train agent
                    training_results = self._train_agent_detailed(
                        agent, env, num_episodes
                    )

                    # Evaluate agent
                    eval_results = self._evaluate_agent_detailed(
                        agent, env, num_episodes=100
                    )

                    multi_env_results[env_name][agent_name] = {
                        "training": training_results,
                        "evaluation": eval_results,
                    }

                    env.close()

                except Exception as e:
                    print(f"    Error with {agent_name} on {env_name}: {e}")
                    multi_env_results[env_name][agent_name] = {"error": str(e)}

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.base_dir, f"multi_env_study_{timestamp}.json")

        with open(results_file, "w") as f:
            json.dump(multi_env_results, f, indent=2)

        # Generate multi-environment analysis
        self._generate_multi_env_analysis(multi_env_results, timestamp)

        return multi_env_results

    def run_scalability_study(
        self, agent_class, env_sizes: List[int], num_episodes: int = 1000
    ) -> Dict[str, Any]:
        """Study how agents scale with environment complexity"""

        print(f"Starting scalability study with {len(env_sizes)} environment sizes...")

        scalability_results = {}

        for size in env_sizes:
            print(f"\nTesting environment size: {size}x{size}")

            try:
                # Create custom environment with specified size
                from environments.custom_envs import GridWorldEnv

                env = GridWorldEnv(size=size)

                # Create agent
                agent = agent_class(state_dim=size * size, action_dim=4)

                # Train agent
                training_results = self._train_agent_detailed(agent, env, num_episodes)

                # Evaluate agent
                eval_results = self._evaluate_agent_detailed(
                    agent, env, num_episodes=100
                )

                scalability_results[f"size_{size}"] = {
                    "training": training_results,
                    "evaluation": eval_results,
                    "environment_size": size,
                    "state_space_size": size * size,
                }

                env.close()

            except Exception as e:
                print(f"    Error with size {size}: {e}")
                scalability_results[f"size_{size}"] = {"error": str(e)}

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            self.base_dir, f"scalability_study_{timestamp}.json"
        )

        with open(results_file, "w") as f:
            json.dump(scalability_results, f, indent=2)

        # Generate scalability analysis
        self._generate_scalability_analysis(scalability_results, timestamp)

        return scalability_results

    def run_hyperparameter_sensitivity_study(
        self,
        agent_class,
        param_ranges: Dict[str, Tuple],
        env_name: str = "CartPole-v1",
        num_episodes: int = 500,
    ) -> Dict[str, Any]:
        """Study sensitivity to hyperparameter changes"""

        print(f"Starting hyperparameter sensitivity study...")

        sensitivity_results = {}

        for param_name, (min_val, max_val, num_points) in param_ranges.items():
            print(f"\nTesting parameter: {param_name}")

            param_values = np.linspace(min_val, max_val, num_points)
            param_results = []

            for value in param_values:
                print(f"  Testing {param_name} = {value:.4f}")

                try:
                    # Create environment
                    env = gym.make(env_name)

                    # Create agent with specific parameter value
                    agent_params = {param_name: value}
                    agent = agent_class(
                        state_dim=env.observation_space.shape[0],
                        action_dim=env.action_space.n,
                        **agent_params,
                    )

                    # Train agent
                    training_results = self._train_agent_detailed(
                        agent, env, num_episodes
                    )

                    # Evaluate agent
                    eval_results = self._evaluate_agent_detailed(
                        agent, env, num_episodes=50
                    )

                    param_results.append(
                        {
                            "parameter_value": value,
                            "training": training_results,
                            "evaluation": eval_results,
                        }
                    )

                    env.close()

                except Exception as e:
                    print(f"    Error with {param_name} = {value}: {e}")
                    param_results.append({"parameter_value": value, "error": str(e)})

            sensitivity_results[param_name] = param_results

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            self.base_dir, f"sensitivity_study_{timestamp}.json"
        )

        with open(results_file, "w") as f:
            json.dump(sensitivity_results, f, indent=2)

        # Generate sensitivity analysis
        self._generate_sensitivity_analysis(sensitivity_results, timestamp)

        return sensitivity_results

    def run_parallel_experiments(
        self, experiment_configs: List[Dict], max_workers: int = None
    ) -> Dict[str, Any]:
        """Run multiple experiments in parallel"""

        if max_workers is None:
            max_workers = min(len(experiment_configs), mp.cpu_count())

        print(
            f"Running {len(experiment_configs)} experiments in parallel with {max_workers} workers..."
        )

        # Prepare experiment functions
        experiment_functions = []
        for config in experiment_configs:
            experiment_functions.append(self._run_single_experiment)

        # Run experiments in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(self._run_single_experiment, experiment_configs)
            )

        # Organize results
        parallel_results = {}
        for i, (config, result) in enumerate(zip(experiment_configs, results)):
            experiment_name = config.get("name", f"experiment_{i}")
            parallel_results[experiment_name] = result

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            self.base_dir, f"parallel_experiments_{timestamp}.json"
        )

        with open(results_file, "w") as f:
            json.dump(parallel_results, f, indent=2)

        return parallel_results

    def _train_agent_detailed(self, agent, env, num_episodes: int) -> Dict[str, Any]:
        """Train agent with detailed metrics collection"""

        episode_rewards = []
        episode_lengths = []
        losses = []
        q_values_history = []
        epsilon_history = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                agent.replay_buffer.push(state, action, reward, next_state, done)

                if len(agent.replay_buffer) > agent.batch_size:
                    loss = agent.update()
                    losses.append(loss)

                episode_reward += reward
                episode_length += 1
                state = next_state

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Record Q-values periodically
            if episode % 100 == 0:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_vals = agent.q_network(state_tensor).numpy()
                    q_values_history.append(q_vals[0].tolist())

            # Record epsilon
            if hasattr(agent, "epsilon"):
                epsilon_history.append(agent.epsilon)
                agent.epsilon = max(
                    agent.epsilon_end, agent.epsilon * agent.epsilon_decay
                )

            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"    Episode {episode}, Average Reward: {avg_reward:.2f}")

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "losses": losses,
            "q_values_history": q_values_history,
            "epsilon_history": epsilon_history,
            "final_avg_reward": (
                np.mean(episode_rewards[-100:])
                if len(episode_rewards) >= 100
                else np.mean(episode_rewards)
            ),
            "total_episodes": num_episodes,
        }

    def _evaluate_agent_detailed(
        self, agent, env, num_episodes: int = 100
    ) -> Dict[str, Any]:
        """Evaluate agent with detailed metrics"""

        episode_rewards = []
        episode_lengths = []
        success_rate = 0

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = agent.select_action(state, epsilon=0.0)  # No exploration
                next_state, reward, done, info = env.step(action)

                episode_reward += reward
                episode_length += 1
                state = next_state

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Check success (customize based on environment)
            if episode_reward > 0:
                success_rate += 1

        success_rate /= num_episodes

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "success_rate": success_rate,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "evaluation_episodes": num_episodes,
        }

    def _aggregate_results(self, run_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across multiple runs"""

        # Extract metrics
        final_rewards = [run["training"]["final_avg_reward"] for run in run_results]
        eval_rewards = [run["evaluation"]["mean_reward"] for run in run_results]
        success_rates = [run["evaluation"]["success_rate"] for run in run_results]

        return {
            "num_runs": len(run_results),
            "final_reward_mean": np.mean(final_rewards),
            "final_reward_std": np.std(final_rewards),
            "eval_reward_mean": np.mean(eval_rewards),
            "eval_reward_std": np.std(eval_rewards),
            "success_rate_mean": np.mean(success_rates),
            "success_rate_std": np.std(success_rates),
            "individual_runs": run_results,
        }

    def _run_single_experiment(self, config: Dict) -> Dict[str, Any]:
        """Run a single experiment (for parallel execution)"""

        try:
            agent_class = config["agent_class"]
            env_name = config["env_name"]
            agent_params = config.get("agent_params", {})
            num_episodes = config.get("num_episodes", 1000)

            # Create environment
            env = gym.make(env_name)

            # Create agent
            agent = agent_class(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                **agent_params,
            )

            # Train agent
            training_results = self._train_agent_detailed(agent, env, num_episodes)

            # Evaluate agent
            eval_results = self._evaluate_agent_detailed(agent, env, num_episodes=100)

            env.close()

            return {
                "success": True,
                "training": training_results,
                "evaluation": eval_results,
                "config": config,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "config": config}

    def _generate_ablation_analysis(self, results: Dict[str, Any], timestamp: str):
        """Generate ablation study analysis and visualizations"""

        # Create ablation analysis plot
        config_names = list(results.keys())
        final_rewards = [
            results[config]["final_reward_mean"] for config in config_names
        ]
        eval_rewards = [results[config]["eval_reward_mean"] for config in config_names]
        success_rates = [
            results[config]["success_rate_mean"] for config in config_names
        ]

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Final Training Reward",
                "Evaluation Reward",
                "Success Rate",
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
        )

        fig.add_trace(
            go.Bar(x=config_names, y=final_rewards, name="Final Training Reward"),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(x=config_names, y=eval_rewards, name="Evaluation Reward"),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Bar(x=config_names, y=success_rates, name="Success Rate"), row=1, col=3
        )

        fig.update_layout(title="Ablation Study Results", height=400, showlegend=False)

        # Save plot
        plot_file = os.path.join(self.base_dir, f"ablation_analysis_{timestamp}.html")
        fig.write_html(plot_file)

        print(f"Ablation analysis saved to: {plot_file}")

    def _generate_multi_env_analysis(self, results: Dict[str, Any], timestamp: str):
        """Generate multi-environment analysis"""

        # Create environment comparison plot
        env_names = list(results.keys())

        fig = go.Figure()

        for env_name in env_names:
            if "error" not in results[env_name]:
                agent_names = list(results[env_name].keys())
                eval_rewards = [
                    results[env_name][agent]["evaluation"]["mean_reward"]
                    for agent in agent_names
                    if "evaluation" in results[env_name][agent]
                ]

                fig.add_trace(go.Bar(name=env_name, x=agent_names, y=eval_rewards))

        fig.update_layout(
            title="Multi-Environment Performance Comparison",
            xaxis_title="Agent",
            yaxis_title="Mean Evaluation Reward",
            barmode="group",
        )

        # Save plot
        plot_file = os.path.join(self.base_dir, f"multi_env_analysis_{timestamp}.html")
        fig.write_html(plot_file)

        print(f"Multi-environment analysis saved to: {plot_file}")

    def _generate_scalability_analysis(self, results: Dict[str, Any], timestamp: str):
        """Generate scalability analysis"""

        sizes = []
        rewards = []
        state_spaces = []

        for size_key, result in results.items():
            if "error" not in result:
                sizes.append(result["environment_size"])
                rewards.append(result["evaluation"]["mean_reward"])
                state_spaces.append(result["state_space_size"])

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Performance vs Environment Size",
                "Performance vs State Space Size",
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]],
        )

        fig.add_trace(
            go.Scatter(x=sizes, y=rewards, mode="lines+markers", name="Reward vs Size"),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=state_spaces,
                y=rewards,
                mode="lines+markers",
                name="Reward vs State Space",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(title="Scalability Analysis", height=400)

        # Save plot
        plot_file = os.path.join(
            self.base_dir, f"scalability_analysis_{timestamp}.html"
        )
        fig.write_html(plot_file)

        print(f"Scalability analysis saved to: {plot_file}")

    def _generate_sensitivity_analysis(self, results: Dict[str, Any], timestamp: str):
        """Generate sensitivity analysis"""

        num_params = len(results)
        fig = make_subplots(
            rows=1,
            cols=num_params,
            subplot_titles=list(results.keys()),
            specs=[[{"type": "scatter"}] * num_params],
        )

        for i, (param_name, param_results) in enumerate(results.items()):
            values = [r["parameter_value"] for r in param_results if "error" not in r]
            rewards = [
                r["evaluation"]["mean_reward"]
                for r in param_results
                if "error" not in r
            ]

            fig.add_trace(
                go.Scatter(x=values, y=rewards, mode="lines+markers", name=param_name),
                row=1,
                col=i + 1,
            )

        fig.update_layout(title="Hyperparameter Sensitivity Analysis", height=400)

        # Save plot
        plot_file = os.path.join(
            self.base_dir, f"sensitivity_analysis_{timestamp}.html"
        )
        fig.write_html(plot_file)

        print(f"Sensitivity analysis saved to: {plot_file}")


class ComprehensiveEvaluator:
    """Comprehensive evaluation framework"""

    def __init__(self):
        self.evaluation_results = {}

    def evaluate_robustness(
        self,
        agent_class,
        env_name: str = "CartPole-v1",
        num_episodes: int = 1000,
        num_runs: int = 10,
    ) -> Dict[str, Any]:
        """Evaluate agent robustness across multiple runs"""

        print(f"Evaluating robustness with {num_runs} runs...")

        run_results = []

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")

            # Create environment with different random seeds
            env = gym.make(env_name)
            env.seed(run)

            # Create agent
            agent = agent_class(
                state_dim=env.observation_space.shape[0], action_dim=env.action_space.n
            )

            # Train agent
            episode_rewards = []
            for episode in range(num_episodes):
                state = env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, info = env.step(action)

                    agent.replay_buffer.push(state, action, reward, next_state, done)

                    if len(agent.replay_buffer) > agent.batch_size:
                        agent.update()

                    episode_reward += reward
                    state = next_state

                episode_rewards.append(episode_reward)

                # Update epsilon
                if hasattr(agent, "epsilon"):
                    agent.epsilon = max(
                        agent.epsilon_end, agent.epsilon * agent.epsilon_decay
                    )

            # Evaluate agent
            eval_rewards = []
            for episode in range(100):
                state = env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action = agent.select_action(state, epsilon=0.0)
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward
                    state = next_state

                eval_rewards.append(episode_reward)

            run_results.append(
                {
                    "run_id": run,
                    "final_training_reward": np.mean(episode_rewards[-100:]),
                    "eval_reward_mean": np.mean(eval_rewards),
                    "eval_reward_std": np.std(eval_rewards),
                    "training_rewards": episode_rewards,
                    "eval_rewards": eval_rewards,
                }
            )

            env.close()

        # Calculate robustness metrics
        final_rewards = [r["final_training_reward"] for r in run_results]
        eval_means = [r["eval_reward_mean"] for r in run_results]

        robustness_metrics = {
            "mean_performance": np.mean(eval_means),
            "std_performance": np.std(eval_means),
            "coefficient_of_variation": np.std(eval_means) / np.mean(eval_means),
            "min_performance": np.min(eval_means),
            "max_performance": np.max(eval_means),
            "performance_range": np.max(eval_means) - np.min(eval_means),
            "consistency_score": 1 - (np.std(eval_means) / np.mean(eval_means)),
            "individual_runs": run_results,
        }

        return robustness_metrics

    def evaluate_generalization(
        self,
        agent_class,
        training_env: str = "CartPole-v1",
        test_envs: List[str] = None,
        num_episodes: int = 1000,
    ) -> Dict[str, Any]:
        """Evaluate agent generalization across different environments"""

        if test_envs is None:
            test_envs = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]

        print(f"Evaluating generalization across {len(test_envs)} environments...")

        # Train agent on training environment
        print(f"Training on {training_env}...")
        train_env = gym.make(training_env)

        agent = agent_class(
            state_dim=train_env.observation_space.shape[0],
            action_dim=train_env.action_space.n,
        )

        # Train agent
        for episode in range(num_episodes):
            state = train_env.reset()
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = train_env.step(action)

                agent.replay_buffer.push(state, action, reward, next_state, done)

                if len(agent.replay_buffer) > agent.batch_size:
                    agent.update()

                state = next_state

            # Update epsilon
            if hasattr(agent, "epsilon"):
                agent.epsilon = max(
                    agent.epsilon_end, agent.epsilon * agent.epsilon_decay
                )

        train_env.close()

        # Test on different environments
        generalization_results = {}

        for test_env_name in test_envs:
            print(f"Testing on {test_env_name}...")

            try:
                test_env = gym.make(test_env_name)

                # Adapt agent if needed (simple approach)
                if test_env.observation_space.shape[0] != agent.state_dim:
                    print(f"  Adapting agent for {test_env_name}...")
                    # Create new agent with correct dimensions
                    agent = agent_class(
                        state_dim=test_env.observation_space.shape[0],
                        action_dim=test_env.action_space.n,
                    )

                # Evaluate agent
                eval_rewards = []
                for episode in range(100):
                    state = test_env.reset()
                    episode_reward = 0
                    done = False

                    while not done:
                        action = agent.select_action(state, epsilon=0.0)
                        next_state, reward, done, info = test_env.step(action)
                        episode_reward += reward
                        state = next_state

                    eval_rewards.append(episode_reward)

                generalization_results[test_env_name] = {
                    "mean_reward": np.mean(eval_rewards),
                    "std_reward": np.std(eval_rewards),
                    "eval_rewards": eval_rewards,
                }

                test_env.close()

            except Exception as e:
                print(f"  Error testing on {test_env_name}: {e}")
                generalization_results[test_env_name] = {"error": str(e)}

        return {
            "training_environment": training_env,
            "test_environments": generalization_results,
            "generalization_score": np.mean(
                [
                    r["mean_reward"]
                    for r in generalization_results.values()
                    if "error" not in r
                ]
            ),
        }


if __name__ == "__main__":
    print("Complex experiments and deep comparisons loaded successfully!")
    print("Available tools:")
    print("- ComplexExperimentRunner: Advanced experiment management")
    print("- ComprehensiveEvaluator: Deep evaluation framework")
