"""
Advanced Analysis and Visualization Tools for DQN Methods
Includes: Complex visualizations, hyperparameter analysis, and deep insights
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from collections import defaultdict, deque
import json
import os
from datetime import datetime


class AdvancedVisualizer:
    """Advanced visualization tools for DQN analysis"""

    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def plot_q_value_heatmap_3d(self, agent, env, save_path: Optional[str] = None):
        """Create 3D Q-value heatmap"""
        if hasattr(env, "size"):
            size = env.size
        else:
            size = 10

        # Create state space
        states = []
        q_values = []

        for i in range(size):
            for j in range(size):
                state = np.zeros((size, size), dtype=np.float32)
                state[i, j] = 1.0
                states.append([i, j])

                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
                    q_vals = agent.q_network(state_tensor).numpy()
                    q_values.append(q_vals[0])

        states = np.array(states)
        q_values = np.array(q_values)

        # Create 3D plot
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=states[:, 0],
                    y=states[:, 1],
                    z=q_values[:, 0],  # Q-value for first action
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=q_values[:, 0],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Q-Value"),
                    ),
                    text=[
                        f"State: ({s[0]}, {s[1]})<br>Q-Value: {q:.3f}"
                        for s, q in zip(states, q_values[:, 0])
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title="3D Q-Value Landscape",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Q-Value",
            ),
            width=800,
            height=600,
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(f"{self.save_dir}/q_value_3d_heatmap.html")

        return fig

    def plot_learning_curves_comparison(
        self, results: Dict[str, Any], save_path: Optional[str] = None
    ):
        """Compare learning curves of different agents"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Episode Rewards",
                "Episode Lengths",
                "Loss Curves",
                "Success Rate",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        colors = px.colors.qualitative.Set1

        for i, (agent_name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]

            # Episode rewards
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(data["episode_rewards"]))),
                    y=data["episode_rewards"],
                    mode="lines",
                    name=f"{agent_name} Rewards",
                    line=dict(color=color),
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )

            # Episode lengths
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(data["episode_lengths"]))),
                    y=data["episode_lengths"],
                    mode="lines",
                    name=f"{agent_name} Lengths",
                    line=dict(color=color),
                    opacity=0.7,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            # Loss curves
            if "losses" in data and data["losses"]:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(data["losses"]))),
                        y=data["losses"],
                        mode="lines",
                        name=f"{agent_name} Loss",
                        line=dict(color=color),
                        opacity=0.7,
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

            # Success rate (rolling average)
            if "episode_rewards" in data:
                window_size = min(100, len(data["episode_rewards"]) // 10)
                if window_size > 0:
                    success_rate = (
                        pd.Series(data["episode_rewards"])
                        .rolling(window_size)
                        .apply(lambda x: (x > 0).mean())
                        .fillna(0)
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(success_rate))),
                            y=success_rate,
                            mode="lines",
                            name=f"{agent_name} Success Rate",
                            line=dict(color=color),
                            opacity=0.7,
                            showlegend=False,
                        ),
                        row=2,
                        col=2,
                    )

        fig.update_layout(
            title="Comprehensive Learning Analysis", height=800, showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(f"{self.save_dir}/learning_curves_comparison.html")

        return fig

    def plot_hyperparameter_sensitivity(
        self, results: Dict[str, Any], save_path: Optional[str] = None
    ):
        """Plot hyperparameter sensitivity analysis"""
        # Extract hyperparameters and performance
        data = []

        for config_name, result in results.items():
            if "config" in result and "evaluation_results" in result:
                config = result["config"]
                perf = result["evaluation_results"]

                data.append(
                    {
                        "config": config_name,
                        "learning_rate": config.get("agent", {}).get("lr", 0.001),
                        "gamma": config.get("agent", {}).get("gamma", 0.99),
                        "epsilon_start": config.get("agent", {}).get(
                            "epsilon_start", 1.0
                        ),
                        "buffer_size": config.get("agent", {}).get(
                            "buffer_size", 10000
                        ),
                        "batch_size": config.get("agent", {}).get("batch_size", 32),
                        "mean_reward": perf.get("mean_reward", 0),
                        "success_rate": perf.get("success_rate", 0),
                    }
                )

        if not data:
            print("No hyperparameter data available")
            return None

        df = pd.DataFrame(data)

        # Create subplots for different hyperparameters
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "Learning Rate",
                "Gamma",
                "Epsilon Start",
                "Buffer Size",
                "Batch Size",
                "Performance Summary",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
            ],
        )

        # Learning Rate vs Performance
        fig.add_trace(
            go.Scatter(
                x=df["learning_rate"],
                y=df["mean_reward"],
                mode="markers+lines",
                name="Mean Reward",
                marker=dict(size=10, color=df["mean_reward"], colorscale="Viridis"),
                text=df["config"],
                hovertemplate="Config: %{text}<br>LR: %{x}<br>Reward: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Gamma vs Performance
        fig.add_trace(
            go.Scatter(
                x=df["gamma"],
                y=df["mean_reward"],
                mode="markers+lines",
                name="Mean Reward",
                marker=dict(size=10, color=df["mean_reward"], colorscale="Viridis"),
                text=df["config"],
                hovertemplate="Config: %{text}<br>Gamma: %{x}<br>Reward: %{y}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Epsilon Start vs Performance
        fig.add_trace(
            go.Scatter(
                x=df["epsilon_start"],
                y=df["mean_reward"],
                mode="markers+lines",
                name="Mean Reward",
                marker=dict(size=10, color=df["mean_reward"], colorscale="Viridis"),
                text=df["config"],
                hovertemplate="Config: %{text}<br>Epsilon: %{x}<br>Reward: %{y}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        # Buffer Size vs Performance
        fig.add_trace(
            go.Scatter(
                x=df["buffer_size"],
                y=df["mean_reward"],
                mode="markers+lines",
                name="Mean Reward",
                marker=dict(size=10, color=df["mean_reward"], colorscale="Viridis"),
                text=df["config"],
                hovertemplate="Config: %{text}<br>Buffer: %{x}<br>Reward: %{y}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Batch Size vs Performance
        fig.add_trace(
            go.Scatter(
                x=df["batch_size"],
                y=df["mean_reward"],
                mode="markers+lines",
                name="Mean Reward",
                marker=dict(size=10, color=df["mean_reward"], colorscale="Viridis"),
                text=df["config"],
                hovertemplate="Config: %{text}<br>Batch: %{x}<br>Reward: %{y}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # Performance Summary
        fig.add_trace(
            go.Bar(
                x=df["config"],
                y=df["mean_reward"],
                name="Mean Reward",
                marker_color=df["mean_reward"],
                text=df["mean_reward"].round(2),
                textposition="auto",
            ),
            row=2,
            col=3,
        )

        fig.update_layout(
            title="Hyperparameter Sensitivity Analysis", height=800, showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(f"{self.save_dir}/hyperparameter_sensitivity.html")

        return fig

    def plot_network_architecture_analysis(
        self, agent, save_path: Optional[str] = None
    ):
        """Analyze and visualize network architecture"""
        # Extract network information
        network_info = {"layers": [], "parameters": 0, "weights": [], "biases": []}

        for name, module in agent.q_network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_info = {
                    "name": name,
                    "type": type(module).__name__,
                    "input_size": (
                        module.in_features if hasattr(module, "in_features") else "N/A"
                    ),
                    "output_size": (
                        module.out_features
                        if hasattr(module, "out_features")
                        else "N/A"
                    ),
                    "parameters": sum(p.numel() for p in module.parameters()),
                }
                network_info["layers"].append(layer_info)
                network_info["parameters"] += layer_info["parameters"]

                # Extract weights and biases
                if hasattr(module, "weight") and module.weight is not None:
                    network_info["weights"].append(module.weight.detach().numpy())
                if hasattr(module, "bias") and module.bias is not None:
                    network_info["biases"].append(module.bias.detach().numpy())

        # Create visualization
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Network Architecture",
                "Weight Distributions",
                "Bias Distributions",
                "Parameter Count",
            ),
            specs=[
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "histogram"}, {"type": "bar"}],
            ],
        )

        # Network Architecture
        layer_names = [layer["name"] for layer in network_info["layers"]]
        param_counts = [layer["parameters"] for layer in network_info["layers"]]

        fig.add_trace(
            go.Bar(
                x=layer_names,
                y=param_counts,
                name="Parameters per Layer",
                marker_color="lightblue",
            ),
            row=1,
            col=1,
        )

        # Weight Distributions
        all_weights = np.concatenate([w.flatten() for w in network_info["weights"]])
        fig.add_trace(
            go.Histogram(
                x=all_weights, name="Weight Distribution", nbinsx=50, opacity=0.7
            ),
            row=1,
            col=2,
        )

        # Bias Distributions
        if network_info["biases"]:
            all_biases = np.concatenate([b.flatten() for b in network_info["biases"]])
            fig.add_trace(
                go.Histogram(
                    x=all_biases, name="Bias Distribution", nbinsx=50, opacity=0.7
                ),
                row=2,
                col=1,
            )

        # Parameter Count Summary
        fig.add_trace(
            go.Bar(
                x=["Total Parameters"],
                y=[network_info["parameters"]],
                name="Total Parameters",
                marker_color="red",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=f"Network Architecture Analysis (Total: {network_info['parameters']:,} parameters)",
            height=800,
            showlegend=True,
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(f"{self.save_dir}/network_architecture_analysis.html")

        return fig

    def plot_experience_replay_analysis(
        self, replay_buffer, save_path: Optional[str] = None
    ):
        """Advanced analysis of experience replay buffer"""
        if len(replay_buffer) == 0:
            print("Replay buffer is empty")
            return None

        # Extract experiences
        experiences = list(replay_buffer)
        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])

        # Create visualization
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "Reward Distribution",
                "Action Distribution",
                "Done Distribution",
                "State Space Coverage",
                "Reward vs Action",
                "Experience Timeline",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            ],
        )

        # Reward Distribution
        fig.add_trace(
            go.Histogram(x=rewards, name="Reward Distribution", nbinsx=30, opacity=0.7),
            row=1,
            col=1,
        )

        # Action Distribution
        action_counts = np.bincount(actions)
        fig.add_trace(
            go.Bar(
                x=list(range(len(action_counts))),
                y=action_counts,
                name="Action Distribution",
                marker_color="lightgreen",
            ),
            row=1,
            col=2,
        )

        # Done Distribution
        done_counts = np.bincount(dones.astype(int))
        fig.add_trace(
            go.Pie(
                labels=["Not Done", "Done"],
                values=done_counts,
                name="Episode Completion",
            ),
            row=1,
            col=3,
        )

        # State Space Coverage (if 2D states)
        if states.shape[1] >= 2:
            fig.add_trace(
                go.Scatter(
                    x=states[:, 0],
                    y=states[:, 1],
                    mode="markers",
                    name="State Coverage",
                    marker=dict(
                        size=3, opacity=0.5, color=rewards, colorscale="Viridis"
                    ),
                ),
                row=2,
                col=1,
            )

        # Reward vs Action
        fig.add_trace(
            go.Scatter(
                x=actions,
                y=rewards,
                mode="markers",
                name="Reward vs Action",
                marker=dict(size=5, opacity=0.6),
            ),
            row=2,
            col=2,
        )

        # Experience Timeline
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rewards))),
                y=rewards,
                mode="lines+markers",
                name="Reward Timeline",
                line=dict(width=1),
                marker=dict(size=2),
            ),
            row=2,
            col=3,
        )

        fig.update_layout(
            title=f"Experience Replay Analysis ({len(experiences)} experiences)",
            height=800,
            showlegend=True,
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(f"{self.save_dir}/experience_replay_analysis.html")

        return fig


class HyperparameterOptimizer:
    """Hyperparameter optimization using various strategies"""

    def __init__(self, agent_class, env_name: str = "CartPole-v1"):
        self.agent_class = agent_class
        self.env_name = env_name
        self.results = []

    def grid_search(
        self, param_grid: Dict[str, List], num_episodes: int = 500
    ) -> Dict[str, Any]:
        """Perform grid search over hyperparameters"""
        import itertools

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        best_score = float("-inf")
        best_params = None
        results = []

        print(f"Starting grid search with {len(combinations)} combinations...")

        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))

            print(f"Testing combination {i+1}/{len(combinations)}: {params}")

            # Train agent with these parameters
            score = self._evaluate_params(params, num_episodes)

            results.append({"params": params, "score": score, "combination_id": i})

            if score > best_score:
                best_score = score
                best_params = params

            print(f"Score: {score:.3f}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results,
            "param_grid": param_grid,
        }

    def random_search(
        self,
        param_distributions: Dict[str, Any],
        n_iter: int = 50,
        num_episodes: int = 500,
    ) -> Dict[str, Any]:
        """Perform random search over hyperparameters"""
        import random

        best_score = float("-inf")
        best_params = None
        results = []

        print(f"Starting random search with {n_iter} iterations...")

        for i in range(n_iter):
            # Sample random parameters
            params = {}
            for param_name, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    params[param_name] = random.choice(distribution)
                elif isinstance(distribution, tuple) and len(distribution) == 2:
                    # Uniform distribution
                    params[param_name] = random.uniform(
                        distribution[0], distribution[1]
                    )
                else:
                    params[param_name] = distribution

            print(f"Iteration {i+1}/{n_iter}: {params}")

            # Train agent with these parameters
            score = self._evaluate_params(params, num_episodes)

            results.append({"params": params, "score": score, "iteration": i})

            if score > best_score:
                best_score = score
                best_params = params

            print(f"Score: {score:.3f}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results,
            "n_iter": n_iter,
        }

    def bayesian_optimization(
        self, param_bounds: Dict[str, Tuple], n_iter: int = 30, num_episodes: int = 500
    ) -> Dict[str, Any]:
        """Perform Bayesian optimization (simplified version)"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
        except ImportError:
            print("scikit-optimize not available, falling back to random search")
            return self.random_search(param_bounds, n_iter, num_episodes)

        # Define search space
        dimensions = []
        param_names = list(param_bounds.keys())

        for param_name, bounds in param_bounds.items():
            if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                dimensions.append(Integer(bounds[0], bounds[1], name=param_name))
            else:
                dimensions.append(Real(bounds[0], bounds[1], name=param_name))

        def objective(params):
            param_dict = dict(zip(param_names, params))
            score = self._evaluate_params(param_dict, num_episodes)
            return -score  # Minimize negative score

        print(f"Starting Bayesian optimization with {n_iter} iterations...")

        result = gp_minimize(objective, dimensions, n_calls=n_iter, random_state=42)

        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun

        return {
            "best_params": best_params,
            "best_score": best_score,
            "optimization_result": result,
            "n_iter": n_iter,
        }

    def _evaluate_params(self, params: Dict[str, Any], num_episodes: int) -> float:
        """Evaluate a set of parameters"""
        import gym

        # Create environment
        env = gym.make(self.env_name)

        # Create agent with parameters
        agent = self.agent_class(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            **params,
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

        env.close()

        # Return average reward of last 100 episodes
        return (
            np.mean(episode_rewards[-100:])
            if len(episode_rewards) >= 100
            else np.mean(episode_rewards)
        )


class PerformanceAnalyzer:
    """Advanced performance analysis tools"""

    def __init__(self):
        self.analysis_results = {}

    def analyze_convergence(
        self, episode_rewards: List[float], window_size: int = 100
    ) -> Dict[str, Any]:
        """Analyze convergence properties"""
        rewards = np.array(episode_rewards)

        # Rolling average
        rolling_avg = pd.Series(rewards).rolling(window_size).mean().fillna(0)

        # Convergence metrics
        convergence_analysis = {
            "total_episodes": len(rewards),
            "final_performance": (
                np.mean(rewards[-window_size:])
                if len(rewards) >= window_size
                else np.mean(rewards)
            ),
            "best_performance": np.max(rolling_avg),
            "convergence_episode": None,
            "stability": 0,
            "improvement_rate": 0,
        }

        # Find convergence point (when rolling average stabilizes)
        if len(rolling_avg) > window_size * 2:
            # Look for stability in rolling average
            recent_avg = rolling_avg[-window_size:]
            if len(recent_avg) > 0:
                std_dev = np.std(recent_avg)
                if std_dev < np.mean(recent_avg) * 0.1:  # Less than 10% variation
                    convergence_analysis["convergence_episode"] = (
                        len(rewards) - window_size
                    )
                    convergence_analysis["stability"] = 1 - (
                        std_dev / np.mean(recent_avg)
                    )

        # Calculate improvement rate
        if len(rewards) > window_size:
            early_performance = np.mean(rewards[:window_size])
            late_performance = np.mean(rewards[-window_size:])
            convergence_analysis["improvement_rate"] = (
                late_performance - early_performance
            ) / abs(early_performance)

        return convergence_analysis

    def analyze_sample_efficiency(
        self, episode_rewards: List[float], target_performance: float = 0.8
    ) -> Dict[str, Any]:
        """Analyze sample efficiency"""
        rewards = np.array(episode_rewards)

        # Find episodes to reach target performance
        episodes_to_target = None
        for i, reward in enumerate(rewards):
            if reward >= target_performance:
                episodes_to_target = i + 1
                break

        # Calculate area under curve (AUC)
        auc = np.trapz(rewards) / len(rewards)

        # Calculate learning speed (slope of improvement)
        if len(rewards) > 10:
            x = np.arange(len(rewards))
            slope = np.polyfit(x, rewards, 1)[0]
        else:
            slope = 0

        return {
            "episodes_to_target": episodes_to_target,
            "target_performance": target_performance,
            "auc": auc,
            "learning_speed": slope,
            "sample_efficiency": 1 / episodes_to_target if episodes_to_target else 0,
        }

    def analyze_exploration_efficiency(
        self, actions: List[int], rewards: List[float]
    ) -> Dict[str, Any]:
        """Analyze exploration efficiency"""
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Action diversity
        unique_actions = len(np.unique(actions))
        total_actions = len(actions)
        action_diversity = unique_actions / total_actions

        # Reward-action correlation
        reward_action_corr = (
            np.corrcoef(actions, rewards)[0, 1] if len(actions) > 1 else 0
        )

        # Exploration vs exploitation balance
        # Simple heuristic: high diversity = exploration, high correlation = exploitation
        exploration_score = action_diversity
        exploitation_score = abs(reward_action_corr)

        return {
            "action_diversity": action_diversity,
            "reward_action_correlation": reward_action_corr,
            "exploration_score": exploration_score,
            "exploitation_score": exploitation_score,
            "balance_score": exploration_score * exploitation_score,
        }


if __name__ == "__main__":
    print("Advanced analysis and visualization tools loaded successfully!")
    print("Available tools:")
    print("- AdvancedVisualizer: Complex visualizations")
    print("- HyperparameterOptimizer: Parameter tuning")
    print("- PerformanceAnalyzer: Deep performance analysis")
