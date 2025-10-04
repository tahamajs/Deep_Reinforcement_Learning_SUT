"""
Advanced Visualization Tools

This module contains advanced visualization tools for RL experiments:
- Interactive 3D visualizations
- Real-time training monitoring
- Multi-agent trajectory visualization
- Policy heatmaps and value function plots
- Comparative analysis dashboards
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import time
from collections import deque
import threading
import queue

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class AdvancedVisualizer:
    """Advanced visualization toolkit for RL experiments."""

    def __init__(self):
        self.figures = {}
        self.data_queues = {}
        self.animation_threads = {}

    def create_3d_trajectory_plot(
        self, trajectories, title="3D Trajectory Visualization"
    ):
        """Create 3D trajectory visualization."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

        for i, trajectory in enumerate(trajectories):
            if len(trajectory) > 0:
                traj_array = np.array(trajectory)
                if traj_array.shape[1] >= 3:
                    ax.plot(
                        traj_array[:, 0],
                        traj_array[:, 1],
                        traj_array[:, 2],
                        color=colors[i],
                        alpha=0.7,
                        linewidth=2,
                        label=f"Agent {i+1}",
                    )
                    ax.scatter(
                        traj_array[0, 0],
                        traj_array[0, 1],
                        traj_array[0, 2],
                        color=colors[i],
                        s=100,
                        marker="o",
                        label=f"Start {i+1}",
                    )
                    ax.scatter(
                        traj_array[-1, 0],
                        traj_array[-1, 1],
                        traj_array[-1, 2],
                        color=colors[i],
                        s=100,
                        marker="s",
                        label=f"End {i+1}",
                    )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title(title)
        ax.legend()

        return fig

    def create_policy_heatmap(
        self, policy_function, state_space, action_space, title="Policy Heatmap"
    ):
        """Create policy heatmap visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)

        # Create state grid
        if len(state_space) >= 2:
            x = np.linspace(state_space[0][0], state_space[0][1], 50)
            y = np.linspace(state_space[1][0], state_space[1][1], 50)
            X, Y = np.meshgrid(x, y)

            # Compute policy for each state
            Z = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    state = np.array([X[j, i], Y[j, i]])
                    action_probs = policy_function(state)
                    Z[j, i] = np.argmax(action_probs)  # Most likely action

            # Plot policy heatmap
            im = axes[0, 0].imshow(
                Z,
                extent=[
                    state_space[0][0],
                    state_space[0][1],
                    state_space[1][0],
                    state_space[1][1],
                ],
                aspect="auto",
                origin="lower",
                cmap="viridis",
            )
            axes[0, 0].set_title("Policy Heatmap")
            axes[0, 0].set_xlabel("State Dimension 1")
            axes[0, 0].set_ylabel("State Dimension 2")
            plt.colorbar(im, ax=axes[0, 0])

        # Action distribution
        action_counts = np.zeros(len(action_space))
        for i in range(len(x)):
            for j in range(len(y)):
                state = np.array([X[j, i], Y[j, i]])
                action_probs = policy_function(state)
                action_counts[np.argmax(action_probs)] += 1

        axes[0, 1].bar(range(len(action_space)), action_counts)
        axes[0, 1].set_title("Action Distribution")
        axes[0, 1].set_xlabel("Action")
        axes[0, 1].set_ylabel("Count")

        # Policy entropy
        entropy_map = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                state = np.array([X[j, i], Y[j, i]])
                action_probs = policy_function(state)
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
                entropy_map[j, i] = entropy

        im2 = axes[1, 0].imshow(
            entropy_map,
            extent=[
                state_space[0][0],
                state_space[0][1],
                state_space[1][0],
                state_space[1][1],
            ],
            aspect="auto",
            origin="lower",
            cmap="plasma",
        )
        axes[1, 0].set_title("Policy Entropy")
        axes[1, 0].set_xlabel("State Dimension 1")
        axes[1, 0].set_ylabel("State Dimension 2")
        plt.colorbar(im2, ax=axes[1, 0])

        # Value function (if available)
        if hasattr(policy_function, "get_value"):
            value_map = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    state = np.array([X[j, i], Y[j, i]])
                    value_map[j, i] = policy_function.get_value(state)

            im3 = axes[1, 1].imshow(
                value_map,
                extent=[
                    state_space[0][0],
                    state_space[0][1],
                    state_space[1][0],
                    state_space[1][1],
                ],
                aspect="auto",
                origin="lower",
                cmap="coolwarm",
            )
            axes[1, 1].set_title("Value Function")
            axes[1, 1].set_xlabel("State Dimension 1")
            axes[1, 1].set_ylabel("State Dimension 2")
            plt.colorbar(im3, ax=axes[1, 1])

        plt.tight_layout()
        return fig

    def create_multi_agent_comparison(
        self, agent_results, title="Multi-Agent Performance Comparison"
    ):
        """Create comprehensive multi-agent comparison."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Learning Curves",
                "Final Performance",
                "Sample Efficiency",
                "Success Rate",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        colors = px.colors.qualitative.Set1

        # Learning curves
        for i, (agent_name, results) in enumerate(agent_results.items()):
            episodes = np.arange(len(results["rewards"]))
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=results["rewards"],
                    name=f"{agent_name} Rewards",
                    line=dict(color=colors[i]),
                ),
                row=1,
                col=1,
            )

        # Final performance
        agent_names = list(agent_results.keys())
        final_performances = [
            np.mean(results["rewards"][-50:]) for results in agent_results.values()
        ]
        fig.add_trace(
            go.Bar(
                x=agent_names,
                y=final_performances,
                name="Final Performance",
                marker_color=colors[: len(agent_names)],
            ),
            row=1,
            col=2,
        )

        # Sample efficiency
        sample_efficiencies = []
        for results in agent_results.values():
            rewards = results["rewards"]
            threshold = np.mean(rewards) * 0.8
            episodes_to_threshold = np.where(np.array(rewards) >= threshold)[0]
            if len(episodes_to_threshold) > 0:
                sample_efficiencies.append(episodes_to_threshold[0])
            else:
                sample_efficiencies.append(len(rewards))

        fig.add_trace(
            go.Bar(
                x=agent_names,
                y=sample_efficiencies,
                name="Sample Efficiency",
                marker_color=colors[: len(agent_names)],
            ),
            row=2,
            col=1,
        )

        # Success rate
        success_rates = []
        for results in agent_results.values():
            if "successes" in results:
                success_rates.append(np.mean(results["successes"]))
            else:
                success_rates.append(0.0)

        fig.add_trace(
            go.Bar(
                x=agent_names,
                y=success_rates,
                name="Success Rate",
                marker_color=colors[: len(agent_names)],
            ),
            row=2,
            col=2,
        )

        fig.update_layout(height=800, showlegend=True, title_text=title)
        return fig

    def create_real_time_monitor(self, update_interval=1.0):
        """Create real-time training monitor."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Real-Time Training Monitor", fontsize=16)

        # Initialize plots
        (reward_line,) = axes[0, 0].plot([], [], "b-", linewidth=2)
        (loss_line,) = axes[0, 1].plot([], [], "r-", linewidth=2)
        (success_line,) = axes[1, 0].plot([], [], "g-", linewidth=2)
        (value_line,) = axes[1, 1].plot([], [], "m-", linewidth=2)

        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title("Training Loss")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title("Success Rate")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Success Rate")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title("Value Function")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Value")
        axes[1, 1].grid(True, alpha=0.3)

        # Data storage
        self.monitor_data = {
            "rewards": deque(maxlen=1000),
            "losses": deque(maxlen=1000),
            "successes": deque(maxlen=1000),
            "values": deque(maxlen=1000),
        }

        def update_plots(frame):
            """Update plots with new data."""
            if len(self.monitor_data["rewards"]) > 0:
                episodes = np.arange(len(self.monitor_data["rewards"]))

                # Update reward plot
                reward_line.set_data(episodes, list(self.monitor_data["rewards"]))
                axes[0, 0].relim()
                axes[0, 0].autoscale_view()

                # Update loss plot
                if len(self.monitor_data["losses"]) > 0:
                    loss_steps = np.arange(len(self.monitor_data["losses"]))
                    loss_line.set_data(loss_steps, list(self.monitor_data["losses"]))
                    axes[0, 1].relim()
                    axes[0, 1].autoscale_view()

                # Update success plot
                if len(self.monitor_data["successes"]) > 0:
                    success_line.set_data(
                        episodes, list(self.monitor_data["successes"])
                    )
                    axes[1, 0].relim()
                    axes[1, 0].autoscale_view()

                # Update value plot
                if len(self.monitor_data["values"]) > 0:
                    value_line.set_data(episodes, list(self.monitor_data["values"]))
                    axes[1, 1].relim()
                    axes[1, 1].autoscale_view()

        # Create animation
        anim = FuncAnimation(
            fig, update_plots, interval=update_interval * 1000, blit=False
        )

        return fig, anim

    def create_hierarchical_visualization(
        self, hierarchy_data, title="Hierarchical RL Visualization"
    ):
        """Create hierarchical RL visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16)

        # Level-wise performance
        levels = list(hierarchy_data.keys())
        level_performances = [hierarchy_data[level]["performance"] for level in levels]

        axes[0, 0].bar(levels, level_performances, color="skyblue", alpha=0.7)
        axes[0, 0].set_title("Performance by Level")
        axes[0, 0].set_xlabel("Hierarchy Level")
        axes[0, 0].set_ylabel("Performance")
        axes[0, 0].grid(True, alpha=0.3)

        # Subgoal achievement
        subgoal_achievements = [
            hierarchy_data[level]["subgoal_rate"] for level in levels
        ]
        axes[0, 1].plot(
            levels, subgoal_achievements, "o-", color="green", linewidth=2, markersize=8
        )
        axes[0, 1].set_title("Subgoal Achievement Rate")
        axes[0, 1].set_xlabel("Hierarchy Level")
        axes[0, 1].set_ylabel("Achievement Rate")
        axes[0, 1].grid(True, alpha=0.3)

        # Learning curves for each level
        for level in levels:
            if "learning_curve" in hierarchy_data[level]:
                episodes = np.arange(len(hierarchy_data[level]["learning_curve"]))
                axes[1, 0].plot(
                    episodes,
                    hierarchy_data[level]["learning_curve"],
                    label=f"Level {level}",
                    linewidth=2,
                )

        axes[1, 0].set_title("Learning Curves by Level")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Reward")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Hierarchy structure visualization
        y_positions = np.arange(len(levels))
        for i, level in enumerate(levels):
            # Draw level box
            rect = patches.Rectangle(
                (0, i - 0.4),
                1,
                0.8,
                linewidth=2,
                edgecolor="black",
                facecolor="lightblue",
                alpha=0.7,
            )
            axes[1, 1].add_patch(rect)

            # Add level text
            axes[1, 1].text(
                0.5, i, f"Level {level}", ha="center", va="center", fontweight="bold"
            )

            # Draw connections to next level
            if i < len(levels) - 1:
                axes[1, 1].arrow(
                    0.5,
                    i + 0.4,
                    0,
                    0.2,
                    head_width=0.05,
                    head_length=0.05,
                    fc="red",
                    ec="red",
                )

        axes[1, 1].set_xlim(-0.2, 1.2)
        axes[1, 1].set_ylim(-0.5, len(levels) - 0.5)
        axes[1, 1].set_title("Hierarchy Structure")
        axes[1, 1].set_aspect("equal")

        plt.tight_layout()
        return fig

    def create_model_uncertainty_plot(
        self, model_predictions, true_values, title="Model Uncertainty"
    ):
        """Create model uncertainty visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)

        # Prediction vs true values
        axes[0, 0].scatter(true_values, model_predictions["mean"], alpha=0.6, s=20)
        axes[0, 0].plot(
            [true_values.min(), true_values.max()],
            [true_values.min(), true_values.max()],
            "r--",
            linewidth=2,
        )
        axes[0, 0].set_xlabel("True Values")
        axes[0, 0].set_ylabel("Predicted Values")
        axes[0, 0].set_title("Prediction Accuracy")
        axes[0, 0].grid(True, alpha=0.3)

        # Uncertainty distribution
        uncertainties = model_predictions["std"]
        axes[0, 1].hist(
            uncertainties, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 1].set_xlabel("Prediction Uncertainty")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Uncertainty Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        # Prediction error vs uncertainty
        errors = np.abs(model_predictions["mean"] - true_values)
        axes[1, 0].scatter(uncertainties, errors, alpha=0.6, s=20)
        axes[1, 0].set_xlabel("Prediction Uncertainty")
        axes[1, 0].set_ylabel("Prediction Error")
        axes[1, 0].set_title("Error vs Uncertainty")
        axes[1, 0].grid(True, alpha=0.3)

        # Calibration plot
        sorted_indices = np.argsort(uncertainties)
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_errors = errors[sorted_indices]

        # Compute calibration
        num_bins = 10
        bin_size = len(sorted_uncertainties) // num_bins
        calibration_errors = []
        calibration_uncertainties = []

        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, len(sorted_uncertainties))

            bin_uncertainties = sorted_uncertainties[start_idx:end_idx]
            bin_errors = sorted_errors[start_idx:end_idx]

            calibration_uncertainties.append(np.mean(bin_uncertainties))
            calibration_errors.append(np.mean(bin_errors))

        axes[1, 1].plot(
            calibration_uncertainties,
            calibration_errors,
            "o-",
            linewidth=2,
            markersize=8,
        )
        axes[1, 1].plot(
            [0, max(calibration_uncertainties)],
            [0, max(calibration_uncertainties)],
            "r--",
            linewidth=2,
        )
        axes[1, 1].set_xlabel("Mean Uncertainty")
        axes[1, 1].set_ylabel("Mean Error")
        axes[1, 1].set_title("Model Calibration")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_comparative_dashboard(
        self, experiment_results, title="Comparative Analysis Dashboard"
    ):
        """Create comprehensive comparative dashboard."""
        fig = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=(
                "Learning Curves",
                "Final Performance",
                "Sample Efficiency",
                "Success Rate",
                "Computational Cost",
                "Policy Entropy",
                "Value Function",
                "Action Distribution",
                "Reward Distribution",
            ),
            specs=[
                [
                    {"secondary_y": False},
                    {"secondary_y": False},
                    {"secondary_y": False},
                ],
                [
                    {"secondary_y": False},
                    {"secondary_y": False},
                    {"secondary_y": False},
                ],
                [
                    {"secondary_y": False},
                    {"secondary_y": False},
                    {"secondary_y": False},
                ],
            ],
        )

        colors = px.colors.qualitative.Set1

        for i, (exp_name, results) in enumerate(experiment_results.items()):
            color = colors[i % len(colors)]

            # Learning curves
            episodes = np.arange(len(results["rewards"]))
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=results["rewards"],
                    name=f"{exp_name} Rewards",
                    line=dict(color=color),
                ),
                row=1,
                col=1,
            )

            # Final performance
            fig.add_trace(
                go.Bar(
                    x=[exp_name],
                    y=[np.mean(results["rewards"][-50:])],
                    name=f"{exp_name} Final",
                    marker_color=color,
                ),
                row=1,
                col=2,
            )

            # Sample efficiency
            threshold = np.mean(results["rewards"]) * 0.8
            episodes_to_threshold = np.where(np.array(results["rewards"]) >= threshold)[
                0
            ]
            efficiency = (
                episodes_to_threshold[0]
                if len(episodes_to_threshold) > 0
                else len(results["rewards"])
            )

            fig.add_trace(
                go.Bar(
                    x=[exp_name],
                    y=[efficiency],
                    name=f"{exp_name} Efficiency",
                    marker_color=color,
                ),
                row=1,
                col=3,
            )

            # Success rate
            if "successes" in results:
                success_rate = np.mean(results["successes"])
                fig.add_trace(
                    go.Bar(
                        x=[exp_name],
                        y=[success_rate],
                        name=f"{exp_name} Success",
                        marker_color=color,
                    ),
                    row=2,
                    col=1,
                )

            # Computational cost
            if "computation_time" in results:
                fig.add_trace(
                    go.Bar(
                        x=[exp_name],
                        y=[results["computation_time"]],
                        name=f"{exp_name} Cost",
                        marker_color=color,
                    ),
                    row=2,
                    col=2,
                )

            # Policy entropy
            if "policy_entropy" in results:
                fig.add_trace(
                    go.Scatter(
                        x=episodes,
                        y=results["policy_entropy"],
                        name=f"{exp_name} Entropy",
                        line=dict(color=color),
                    ),
                    row=2,
                    col=3,
                )

            # Value function
            if "values" in results:
                fig.add_trace(
                    go.Scatter(
                        x=episodes,
                        y=results["values"],
                        name=f"{exp_name} Values",
                        line=dict(color=color),
                    ),
                    row=3,
                    col=1,
                )

            # Action distribution
            if "action_distribution" in results:
                actions = list(results["action_distribution"].keys())
                counts = list(results["action_distribution"].values())
                fig.add_trace(
                    go.Bar(
                        x=actions,
                        y=counts,
                        name=f"{exp_name} Actions",
                        marker_color=color,
                    ),
                    row=3,
                    col=2,
                )

            # Reward distribution
            fig.add_trace(
                go.Histogram(
                    x=results["rewards"],
                    name=f"{exp_name} Rewards",
                    marker_color=color,
                    opacity=0.7,
                ),
                row=3,
                col=3,
            )

        fig.update_layout(height=1200, showlegend=True, title_text=title)
        return fig

    def save_visualization(self, fig, filename, format="png", dpi=300):
        """Save visualization to file."""
        if hasattr(fig, "write_html"):  # Plotly figure
            fig.write_html(filename)
        else:  # Matplotlib figure
            fig.savefig(filename, format=format, dpi=dpi, bbox_inches="tight")

        print(f"📊 Visualization saved to: {filename}")

    def update_monitor_data(
        self, rewards=None, losses=None, successes=None, values=None
    ):
        """Update real-time monitor data."""
        if rewards is not None:
            self.monitor_data["rewards"].extend(rewards)
        if losses is not None:
            self.monitor_data["losses"].extend(losses)
        if successes is not None:
            self.monitor_data["successes"].extend(successes)
        if values is not None:
            self.monitor_data["values"].extend(values)


class InteractiveVisualizer:
    """Interactive visualization with user controls."""

    def __init__(self):
        self.figures = {}
        self.controls = {}

    def create_interactive_policy_explorer(
        self, policy_function, state_space, action_space
    ):
        """Create interactive policy explorer."""
        fig = go.Figure()

        # Create state grid
        x = np.linspace(state_space[0][0], state_space[0][1], 50)
        y = np.linspace(state_space[1][0], state_space[1][1], 50)
        X, Y = np.meshgrid(x, y)

        # Compute policy
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                state = np.array([X[j, i], Y[j, i]])
                action_probs = policy_function(state)
                Z[j, i] = np.argmax(action_probs)

        # Add heatmap
        fig.add_trace(go.Heatmap(z=Z, x=x, y=y, colorscale="viridis"))

        # Add interactive features
        fig.update_layout(
            title="Interactive Policy Explorer",
            xaxis_title="State Dimension 1",
            yaxis_title="State Dimension 2",
            updatemenus=[
                dict(
                    buttons=list(
                        [
                            dict(label="Policy", method="restyle", args=["z", [Z]]),
                            dict(
                                label="Entropy",
                                method="restyle",
                                args=[
                                    "z",
                                    [self._compute_entropy(X, Y, policy_function)],
                                ],
                            ),
                            dict(
                                label="Value",
                                method="restyle",
                                args=[
                                    "z",
                                    [self._compute_values(X, Y, policy_function)],
                                ],
                            ),
                        ]
                    ),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.02,
                    yanchor="top",
                ),
            ],
        )

        return fig

    def _compute_entropy(self, X, Y, policy_function):
        """Compute policy entropy."""
        entropy_map = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(len(X[0])):
                state = np.array([X[i, j], Y[i, j]])
                action_probs = policy_function(state)
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
                entropy_map[i, j] = entropy
        return entropy_map

    def _compute_values(self, X, Y, policy_function):
        """Compute value function."""
        value_map = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(len(X[0])):
                state = np.array([X[i, j], Y[i, j]])
                if hasattr(policy_function, "get_value"):
                    value_map[i, j] = policy_function.get_value(state)
                else:
                    value_map[i, j] = 0
        return value_map

