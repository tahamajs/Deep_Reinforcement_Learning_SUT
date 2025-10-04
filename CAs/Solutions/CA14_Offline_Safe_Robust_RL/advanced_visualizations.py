"""
Advanced Visualization and Analysis Tools
ابزارهای پیشرفته تجسم و تحلیل

This module contains advanced visualization tools including:
- Interactive 3D Visualizations
- Real-time Performance Monitoring
- Multi-dimensional Analysis
- Causal Graph Visualization
- Quantum State Visualization
- Federated Learning Dashboard
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import torch
import networkx as nx
from collections import deque
import time
import threading
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""

    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8"
    color_palette: str = "viridis"
    animation_fps: int = 30
    real_time_update: bool = True


class Interactive3DVisualizer:
    """Interactive 3D visualization for RL environments."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.fig = None
        self.ax = None
        self.animation = None
        self.data_history = deque(maxlen=1000)

    def create_3d_environment_plot(self, environment_data):
        """Create 3D plot of environment."""
        plt.style.use(self.config.style)
        self.fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Extract data
        agent_positions = environment_data["agent_positions"]
        target_positions = environment_data["target_positions"]
        obstacle_positions = environment_data["obstacle_positions"]
        reward_history = environment_data["reward_history"]

        # Plot agent trajectory
        if len(agent_positions) > 1:
            x, y, z = zip(*agent_positions)
            self.ax.plot(x, y, z, "b-", linewidth=2, label="Agent Trajectory")
            self.ax.scatter(x[0], y[0], z[0], c="green", s=100, label="Start")
            self.ax.scatter(x[-1], y[-1], z[-1], c="red", s=100, label="End")

        # Plot targets
        if target_positions:
            tx, ty, tz = zip(*target_positions)
            self.ax.scatter(tx, ty, tz, c="gold", s=200, marker="*", label="Targets")

        # Plot obstacles
        if obstacle_positions:
            ox, oy, oz = zip(*obstacle_positions)
            self.ax.scatter(ox, oy, oz, c="black", s=100, marker="s", label="Obstacles")

        # Add reward surface
        if len(reward_history) > 10:
            self._add_reward_surface(reward_history)

        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.set_zlabel("Reward")
        self.ax.set_title("3D Environment Visualization")
        self.ax.legend()

        plt.tight_layout()
        return self.fig

    def _add_reward_surface(self, reward_history):
        """Add reward surface to 3D plot."""
        # Create grid for surface
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)

        # Interpolate rewards
        Z = np.zeros_like(X)
        for i in range(len(reward_history)):
            if i < len(reward_history) - 1:
                Z[i % Z.shape[0], i % Z.shape[1]] = reward_history[i]

        # Plot surface
        self.ax.plot_surface(X, Y, Z, alpha=0.3, cmap="viridis")

    def create_animated_trajectory(self, trajectory_data, save_path=None):
        """Create animated trajectory visualization."""
        plt.style.use(self.config.style)
        self.fig, self.ax = plt.subplots(
            figsize=self.config.figure_size, dpi=self.config.dpi
        )

        # Extract data
        positions = trajectory_data["positions"]
        rewards = trajectory_data["rewards"]
        actions = trajectory_data["actions"]

        # Initialize plot
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect("equal")

        # Create animated elements
        (line,) = self.ax.plot([], [], "b-", linewidth=2, label="Trajectory")
        (point,) = self.ax.plot([], [], "ro", markersize=8, label="Current Position")
        reward_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            verticalalignment="top",
            fontsize=12,
        )

        def animate(frame):
            if frame < len(positions):
                # Update trajectory
                x, y = zip(*positions[: frame + 1])
                line.set_data(x, y)

                # Update current position
                point.set_data([positions[frame][0]], [positions[frame][1]])

                # Update reward text
                reward_text.set_text(
                    f"Step: {frame}\nReward: {rewards[frame]:.3f}\nAction: {actions[frame]}"
                )

            return line, point, reward_text

        # Create animation
        self.animation = animation.FuncAnimation(
            self.fig,
            animate,
            frames=len(positions),
            interval=1000 // self.config.animation_fps,
            blit=True,
        )

        self.ax.legend()
        self.ax.set_title("Animated Agent Trajectory")

        if save_path:
            self.animation.save(
                save_path, writer="pillow", fps=self.config.animation_fps
            )

        return self.animation


class RealTimePerformanceMonitor:
    """Real-time performance monitoring dashboard."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.metrics_history = {
            "rewards": deque(maxlen=1000),
            "losses": deque(maxlen=1000),
            "exploration_rate": deque(maxlen=1000),
            "success_rate": deque(maxlen=1000),
            "episode_length": deque(maxlen=1000),
        }
        self.update_thread = None
        self.running = False

    def start_monitoring(self, update_callback=None):
        """Start real-time monitoring."""
        self.running = True
        if self.config.real_time_update:
            self.update_thread = threading.Thread(
                target=self._update_loop, args=(update_callback,)
            )
            self.update_thread.start()

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.running = False
        if self.update_thread:
            self.update_thread.join()

    def update_metrics(self, metrics):
        """Update metrics with new data."""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

    def _update_loop(self, update_callback):
        """Update loop for real-time monitoring."""
        while self.running:
            if update_callback:
                update_callback(self.metrics_history)
            time.sleep(1.0)  # Update every second

    def create_performance_dashboard(self):
        """Create performance dashboard."""
        plt.style.use(self.config.style)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=self.config.dpi)

        # Plot rewards
        if self.metrics_history["rewards"]:
            axes[0, 0].plot(list(self.metrics_history["rewards"]))
            axes[0, 0].set_title("Reward History")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].grid(True)

        # Plot losses
        if self.metrics_history["losses"]:
            axes[0, 1].plot(list(self.metrics_history["losses"]))
            axes[0, 1].set_title("Loss History")
            axes[0, 1].set_xlabel("Update")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].grid(True)

        # Plot exploration rate
        if self.metrics_history["exploration_rate"]:
            axes[0, 2].plot(list(self.metrics_history["exploration_rate"]))
            axes[0, 2].set_title("Exploration Rate")
            axes[0, 2].set_xlabel("Episode")
            axes[0, 2].set_ylabel("Exploration Rate")
            axes[0, 2].grid(True)

        # Plot success rate
        if self.metrics_history["success_rate"]:
            axes[1, 0].plot(list(self.metrics_history["success_rate"]))
            axes[1, 0].set_title("Success Rate")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Success Rate")
            axes[1, 0].grid(True)

        # Plot episode length
        if self.metrics_history["episode_length"]:
            axes[1, 1].plot(list(self.metrics_history["episode_length"]))
            axes[1, 1].set_title("Episode Length")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Length")
            axes[1, 1].grid(True)

        # Plot correlation heatmap
        if all(self.metrics_history[key] for key in self.metrics_history.keys()):
            data = pd.DataFrame(
                {key: list(values) for key, values in self.metrics_history.items()}
            )
            correlation_matrix = data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=axes[1, 2])
            axes[1, 2].set_title("Metrics Correlation")

        plt.tight_layout()
        return fig


class MultiDimensionalAnalyzer:
    """Multi-dimensional analysis and visualization."""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def create_parallel_coordinates_plot(self, data, labels=None):
        """Create parallel coordinates plot."""
        plt.style.use(self.config.style)
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        # Normalize data
        data_normalized = (data - data.min(axis=0)) / (
            data.max(axis=0) - data.min(axis=0)
        )

        # Create parallel coordinates plot
        for i in range(len(data_normalized)):
            ax.plot(range(len(data_normalized[i])), data_normalized[i], alpha=0.7)

        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)

        ax.set_title("Parallel Coordinates Plot")
        ax.grid(True)

        plt.tight_layout()
        return fig

    def create_radar_chart(self, metrics, labels, title="Performance Radar Chart"):
        """Create radar chart for multi-dimensional metrics."""
        plt.style.use(self.config.style)
        fig, ax = plt.subplots(
            figsize=self.config.figure_size,
            dpi=self.config.dpi,
            subplot_kw=dict(projection="polar"),
        )

        # Number of variables
        N = len(metrics)

        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle

        # Add metrics
        metrics += metrics[:1]  # Complete the circle

        # Plot
        ax.plot(angles, metrics, "o-", linewidth=2, label="Performance")
        ax.fill(angles, metrics, alpha=0.25)

        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)

        ax.set_title(title, size=16, fontweight="bold")
        ax.grid(True)

        plt.tight_layout()
        return fig

    def create_heatmap_analysis(
        self, data, row_labels, col_labels, title="Heatmap Analysis"
    ):
        """Create heatmap analysis."""
        plt.style.use(self.config.style)
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        # Create heatmap
        im = ax.imshow(data, cmap="viridis", aspect="auto")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Value")

        # Add labels
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45)
        ax.set_yticklabels(row_labels)

        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(
                    j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white"
                )

        ax.set_title(title)

        plt.tight_layout()
        return fig


class CausalGraphVisualizer:
    """Causal graph visualization tools."""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def create_causal_graph(self, causal_graph, interventions=None):
        """Create causal graph visualization."""
        plt.style.use(self.config.style)
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        # Create networkx graph
        G = nx.DiGraph()

        # Add nodes and edges
        for node, parents in causal_graph.items():
            G.add_node(node)
            for parent in parents:
                G.add_edge(parent, node)

        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_color="lightblue", node_size=2000, alpha=0.8
        )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, edge_color="gray", arrows=True, arrowsize=20, alpha=0.6
        )

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

        # Highlight interventions
        if interventions:
            intervention_nodes = [node for node, _ in interventions]
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=intervention_nodes,
                node_color="red",
                node_size=2000,
                alpha=0.8,
            )

        ax.set_title("Causal Graph")
        ax.axis("off")

        plt.tight_layout()
        return fig

    def create_intervention_analysis(self, intervention_results):
        """Create intervention analysis visualization."""
        plt.style.use(self.config.style)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=self.config.dpi)

        # Extract data
        interventions = list(intervention_results.keys())
        outcomes = [intervention_results[i]["outcome"] for i in interventions]
        confidence_intervals = [
            intervention_results[i]["confidence"] for i in interventions
        ]

        # Plot intervention effects
        axes[0, 0].bar(interventions, outcomes, yerr=confidence_intervals, capsize=5)
        axes[0, 0].set_title("Intervention Effects")
        axes[0, 0].set_xlabel("Intervention")
        axes[0, 0].set_ylabel("Outcome")
        axes[0, 0].grid(True)

        # Plot confidence intervals
        axes[0, 1].errorbar(
            range(len(interventions)),
            outcomes,
            yerr=confidence_intervals,
            fmt="o",
            capsize=5,
        )
        axes[0, 1].set_title("Confidence Intervals")
        axes[0, 1].set_xlabel("Intervention Index")
        axes[0, 1].set_ylabel("Outcome")
        axes[0, 1].grid(True)

        # Plot distribution of outcomes
        all_outcomes = []
        for result in intervention_results.values():
            all_outcomes.extend(result.get("distribution", [result["outcome"]]))

        axes[1, 0].hist(all_outcomes, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 0].set_title("Outcome Distribution")
        axes[1, 0].set_xlabel("Outcome Value")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True)

        # Plot intervention comparison
        baseline_outcome = intervention_results.get("baseline", {}).get("outcome", 0)
        intervention_effects = [outcome - baseline_outcome for outcome in outcomes]

        axes[1, 1].bar(interventions, intervention_effects)
        axes[1, 1].set_title("Intervention Effects vs Baseline")
        axes[1, 1].set_xlabel("Intervention")
        axes[1, 1].set_ylabel("Effect Size")
        axes[1, 1].grid(True)

        plt.tight_layout()
        return fig


class QuantumStateVisualizer:
    """Quantum state visualization tools."""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def create_bloch_sphere(self, quantum_state):
        """Create Bloch sphere visualization."""
        fig = go.Figure()

        # Convert quantum state to Bloch sphere coordinates
        if isinstance(quantum_state, torch.Tensor):
            quantum_state = quantum_state.detach().cpu().numpy()

        # For 2-qubit system, visualize first qubit
        if len(quantum_state) >= 2:
            # Extract first qubit state
            alpha = quantum_state[0]
            beta = quantum_state[1] if len(quantum_state) > 1 else 0

            # Convert to Bloch sphere coordinates
            x = 2 * np.real(alpha * np.conj(beta))
            y = 2 * np.imag(alpha * np.conj(beta))
            z = np.abs(alpha) ** 2 - np.abs(beta) ** 2

            # Create Bloch sphere
            fig.add_trace(
                go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    name="Quantum State",
                )
            )

        # Add sphere surface
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=x_sphere,
                y=y_sphere,
                z=z_sphere,
                opacity=0.3,
                colorscale="Blues",
                name="Bloch Sphere",
            )
        )

        fig.update_layout(
            title="Bloch Sphere Visualization",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        )

        return fig

    def create_quantum_circuit_diagram(self, gates, qubits):
        """Create quantum circuit diagram."""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.config.dpi)

        # Draw qubit lines
        for i in range(qubits):
            ax.plot([0, 10], [i, i], "k-", linewidth=2)
            ax.text(-0.5, i, f"q{i}", ha="right", va="center", fontsize=12)

        # Draw gates
        gate_positions = {}
        for gate in gates:
            gate_type = gate["type"]
            qubit = gate["qubit"]
            position = gate["position"]

            if gate_type == "X":
                ax.plot(position, qubit, "ro", markersize=15)
                ax.text(
                    position,
                    qubit,
                    "X",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )
            elif gate_type == "Y":
                ax.plot(position, qubit, "go", markersize=15)
                ax.text(
                    position,
                    qubit,
                    "Y",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )
            elif gate_type == "Z":
                ax.plot(position, qubit, "bo", markersize=15)
                ax.text(
                    position,
                    qubit,
                    "Z",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )
            elif gate_type == "H":
                ax.plot(position, qubit, "yo", markersize=15)
                ax.text(
                    position,
                    qubit,
                    "H",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax.set_xlim(-1, 11)
        ax.set_ylim(-0.5, qubits - 0.5)
        ax.set_title("Quantum Circuit Diagram")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Qubits")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class FederatedLearningDashboard:
    """Federated learning dashboard."""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def create_federated_dashboard(self, federated_data):
        """Create federated learning dashboard."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Client Performance",
                "Global vs Local Loss",
                "Communication Rounds",
                "Data Distribution",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "pie"}],
            ],
        )

        # Client performance
        for client_id, performance in federated_data["client_performance"].items():
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(performance))),
                    y=performance,
                    mode="lines",
                    name=f"Client {client_id}",
                    line=dict(width=2),
                ),
                row=1,
                col=1,
            )

        # Global vs local loss
        fig.add_trace(
            go.Scatter(
                x=list(range(len(federated_data["global_loss"]))),
                y=federated_data["global_loss"],
                mode="lines",
                name="Global Loss",
                line=dict(color="red", width=3),
            ),
            row=1,
            col=2,
        )

        # Communication rounds
        rounds = list(federated_data["communication_rounds"].keys())
        round_values = list(federated_data["communication_rounds"].values())

        fig.add_trace(
            go.Bar(
                x=rounds,
                y=round_values,
                name="Communication Rounds",
                marker_color="lightblue",
            ),
            row=2,
            col=1,
        )

        # Data distribution
        data_sizes = list(federated_data["data_distribution"].values())
        client_labels = list(federated_data["data_distribution"].keys())

        fig.add_trace(
            go.Pie(labels=client_labels, values=data_sizes, name="Data Distribution"),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Federated Learning Dashboard", showlegend=True, height=800
        )

        return fig

    def create_privacy_analysis(self, privacy_metrics):
        """Create privacy analysis visualization."""
        plt.style.use(self.config.style)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=self.config.dpi)

        # Differential privacy budget
        if "epsilon_values" in privacy_metrics:
            axes[0, 0].plot(privacy_metrics["epsilon_values"])
            axes[0, 0].set_title("Differential Privacy Budget (ε)")
            axes[0, 0].set_xlabel("Communication Round")
            axes[0, 0].set_ylabel("ε Value")
            axes[0, 0].grid(True)

        # Privacy-utility trade-off
        if "privacy_utility_tradeoff" in privacy_metrics:
            privacy_values = privacy_metrics["privacy_utility_tradeoff"]["privacy"]
            utility_values = privacy_metrics["privacy_utility_tradeoff"]["utility"]
            axes[0, 1].scatter(privacy_values, utility_values, alpha=0.7)
            axes[0, 1].set_title("Privacy-Utility Trade-off")
            axes[0, 1].set_xlabel("Privacy Level")
            axes[0, 1].set_ylabel("Utility Level")
            axes[0, 1].grid(True)

        # Noise analysis
        if "noise_analysis" in privacy_metrics:
            noise_values = privacy_metrics["noise_analysis"]
            axes[1, 0].hist(noise_values, bins=30, alpha=0.7, edgecolor="black")
            axes[1, 0].set_title("Noise Distribution")
            axes[1, 0].set_xlabel("Noise Value")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].grid(True)

        # Privacy leakage over time
        if "privacy_leakage" in privacy_metrics:
            leakage_values = privacy_metrics["privacy_leakage"]
            axes[1, 1].plot(leakage_values)
            axes[1, 1].set_title("Privacy Leakage Over Time")
            axes[1, 1].set_xlabel("Time Step")
            axes[1, 1].set_ylabel("Privacy Leakage")
            axes[1, 1].grid(True)

        plt.tight_layout()
        return fig


class AdvancedMetricsAnalyzer:
    """Advanced metrics analysis and visualization."""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def create_comprehensive_analysis(self, all_results):
        """Create comprehensive analysis dashboard."""
        plt.style.use(self.config.style)
        fig = plt.figure(figsize=(20, 15), dpi=self.config.dpi)

        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. Performance comparison heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_performance_heatmap(all_results, ax1)

        # 2. Learning curves
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_learning_curves(all_results, ax2)

        # 3. Sample efficiency analysis
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_sample_efficiency(all_results, ax3)

        # 4. Robustness analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_robustness_analysis(all_results, ax4)

        # 5. Safety analysis
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_safety_analysis(all_results, ax5)

        # 6. Multi-agent coordination
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_coordination_analysis(all_results, ax6)

        # 7. Computational cost analysis
        ax7 = fig.add_subplot(gs[3, :2])
        self._plot_computational_cost(all_results, ax7)

        # 8. Overall ranking
        ax8 = fig.add_subplot(gs[3, 2:])
        self._plot_overall_ranking(all_results, ax8)

        plt.suptitle(
            "Comprehensive Advanced RL Analysis Dashboard",
            fontsize=20,
            fontweight="bold",
        )
        return fig

    def _plot_performance_heatmap(self, results, ax):
        """Plot performance heatmap."""
        methods = list(results.keys())
        metrics = [
            "Sample Efficiency",
            "Asymptotic Performance",
            "Robustness",
            "Safety",
            "Coordination",
        ]

        # Create performance matrix
        performance_matrix = np.zeros((len(methods), len(metrics)))

        for i, method in enumerate(methods):
            for j, metric in enumerate(metrics):
                if metric in results[method]:
                    performance_matrix[i, j] = results[method][metric]

        # Plot heatmap
        im = ax.imshow(performance_matrix, cmap="viridis", aspect="auto")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Performance Score")

        # Add labels
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(methods)))
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_yticklabels(methods)

        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(metrics)):
                text = ax.text(
                    j,
                    i,
                    f"{performance_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )

        ax.set_title("Performance Heatmap")

    def _plot_learning_curves(self, results, ax):
        """Plot learning curves."""
        for method, data in results.items():
            if "learning_curve" in data:
                ax.plot(data["learning_curve"], label=method, linewidth=2)

        ax.set_title("Learning Curves Comparison")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True)

    def _plot_sample_efficiency(self, results, ax):
        """Plot sample efficiency analysis."""
        methods = []
        efficiency_scores = []

        for method, data in results.items():
            if "sample_efficiency" in data:
                methods.append(method)
                efficiency_scores.append(data["sample_efficiency"])

        bars = ax.bar(methods, efficiency_scores, color="skyblue", edgecolor="navy")
        ax.set_title("Sample Efficiency Comparison")
        ax.set_ylabel("Episodes to Convergence")
        ax.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, score in zip(bars, efficiency_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{score:.1f}",
                ha="center",
                va="bottom",
            )

    def _plot_robustness_analysis(self, results, ax):
        """Plot robustness analysis."""
        methods = []
        robustness_scores = []

        for method, data in results.items():
            if "robustness" in data:
                methods.append(method)
                robustness_scores.append(data["robustness"])

        bars = ax.bar(
            methods, robustness_scores, color="lightcoral", edgecolor="darkred"
        )
        ax.set_title("Robustness Analysis")
        ax.set_ylabel("Robustness Score")
        ax.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, score in zip(bars, robustness_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

    def _plot_safety_analysis(self, results, ax):
        """Plot safety analysis."""
        methods = []
        safety_scores = []
        violation_rates = []

        for method, data in results.items():
            if "safety" in data:
                methods.append(method)
                safety_scores.append(data["safety"])
                violation_rates.append(data.get("violation_rate", 0))

        # Create dual y-axis plot
        ax2 = ax.twinx()

        bars1 = ax.bar(
            [m + "_safety" for m in methods],
            safety_scores,
            color="lightgreen",
            alpha=0.7,
            label="Safety Score",
        )
        bars2 = ax2.bar(
            [m + "_violation" for m in methods],
            violation_rates,
            color="red",
            alpha=0.7,
            label="Violation Rate",
        )

        ax.set_title("Safety Analysis")
        ax.set_ylabel("Safety Score", color="green")
        ax2.set_ylabel("Violation Rate", color="red")
        ax.tick_params(axis="x", rotation=45)

        # Add legends
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    def _plot_coordination_analysis(self, results, ax):
        """Plot coordination analysis."""
        methods = []
        coordination_scores = []

        for method, data in results.items():
            if "coordination" in data:
                methods.append(method)
                coordination_scores.append(data["coordination"])

        bars = ax.bar(methods, coordination_scores, color="gold", edgecolor="orange")
        ax.set_title("Multi-Agent Coordination Analysis")
        ax.set_ylabel("Coordination Score")
        ax.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, score in zip(bars, coordination_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

    def _plot_computational_cost(self, results, ax):
        """Plot computational cost analysis."""
        methods = []
        training_times = []
        memory_usage = []

        for method, data in results.items():
            if "computational_cost" in data:
                methods.append(method)
                training_times.append(
                    data["computational_cost"].get("training_time", 0)
                )
                memory_usage.append(data["computational_cost"].get("memory_usage", 0))

        # Create dual y-axis plot
        ax2 = ax.twinx()

        bars1 = ax.bar(
            [m + "_time" for m in methods],
            training_times,
            color="purple",
            alpha=0.7,
            label="Training Time (s)",
        )
        bars2 = ax2.bar(
            [m + "_memory" for m in methods],
            memory_usage,
            color="brown",
            alpha=0.7,
            label="Memory Usage (MB)",
        )

        ax.set_title("Computational Cost Analysis")
        ax.set_ylabel("Training Time (s)", color="purple")
        ax2.set_ylabel("Memory Usage (MB)", color="brown")
        ax.tick_params(axis="x", rotation=45)

        # Add legends
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    def _plot_overall_ranking(self, results, ax):
        """Plot overall ranking."""
        # Calculate overall scores
        method_scores = {}

        for method, data in results.items():
            score = 0
            count = 0

            for metric in [
                "sample_efficiency",
                "asymptotic_performance",
                "robustness",
                "safety",
                "coordination",
            ]:
                if metric in data:
                    score += data[metric]
                    count += 1

            if count > 0:
                method_scores[method] = score / count

        # Sort by score
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        methods, scores = zip(*sorted_methods)

        # Create horizontal bar plot
        bars = ax.barh(methods, scores, color="steelblue", edgecolor="navy")
        ax.set_title("Overall Performance Ranking")
        ax.set_xlabel("Overall Score")

        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                ha="left",
                va="center",
            )

        # Invert y-axis to show best method at top
        ax.invert_yaxis()
