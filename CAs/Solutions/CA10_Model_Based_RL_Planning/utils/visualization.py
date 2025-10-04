"""
Visualization utilities for Model-Based RL
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import os


def plot_learning_curves(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Learning Curves",
) -> plt.Figure:
    """Plot learning curves for multiple methods"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)

    # Plot episode rewards
    for method_name, method_data in results.items():
        if "episode_rewards" in method_data:
            rewards = method_data["episode_rewards"]
            if isinstance(rewards, list) and len(rewards) > 0:
                # Smooth the curve
                smoothed = pd.Series(rewards).rolling(window=10, min_periods=1).mean()
                axes[0].plot(smoothed, label=method_name, linewidth=2)

    axes[0].set_title("Episode Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot episode lengths
    for method_name, method_data in results.items():
        if "episode_lengths" in method_data:
            lengths = method_data["episode_lengths"]
            if isinstance(lengths, list) and len(lengths) > 0:
                smoothed = pd.Series(lengths).rolling(window=10, min_periods=1).mean()
                axes[1].plot(smoothed, label=method_name, linewidth=2)

    axes[1].set_title("Episode Lengths")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_comparison(
    results: Dict[str, Any],
    metric: str = "final_performance",
    save_path: Optional[str] = None,
    title: str = "Method Comparison",
) -> plt.Figure:
    """Plot comparison of methods for a specific metric"""

    methods = list(results.keys())
    values = []
    errors = []

    for method in methods:
        method_data = results[method]
        if metric in method_data:
            values.append(method_data[metric])
        else:
            values.append(0)

        # Look for error/std values
        error_key = (
            f"std_{metric}" if f"std_{metric}" in method_data else f"{metric}_std"
        )
        if error_key in method_data:
            errors.append(method_data[error_key])
        else:
            errors.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        methods,
        values,
        yerr=errors,
        capsize=5,
        alpha=0.7,
        color=["skyblue", "lightgreen", "lightcoral", "gold", "violet"][: len(methods)],
    )

    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_performance_heatmap(
    results: Dict[str, Any], metrics: List[str] = None, save_path: Optional[str] = None
) -> plt.Figure:
    """Create heatmap of performance metrics"""

    if metrics is None:
        metrics = [
            "final_performance",
            "learning_efficiency",
            "sample_efficiency",
            "stability",
        ]

    methods = list(results.keys())
    data_matrix = []

    for method in methods:
        method_data = results[method]
        row = []
        for metric in metrics:
            if metric in method_data:
                row.append(method_data[metric])
            else:
                row.append(0)
        data_matrix.append(row)

    df = pd.DataFrame(data_matrix, index=methods, columns=metrics)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlOrRd", ax=ax, fmt=".3f")
    ax.set_title("Performance Metrics Heatmap")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_planning_analysis(
    results: Dict[str, Any], save_path: Optional[str] = None
) -> plt.Figure:
    """Plot planning-specific analysis"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Planning Analysis", fontsize=16)

    methods = list(results.keys())

    # Planning steps vs performance
    planning_steps = []
    performances = []

    for method in methods:
        method_data = results[method]
        if "planning_steps" in method_data:
            planning_steps.append(method_data["planning_steps"])
            performances.append(method_data.get("final_performance", 0))

    if planning_steps:
        axes[0, 0].scatter(planning_steps, performances, s=100, alpha=0.7)
        axes[0, 0].set_xlabel("Planning Steps")
        axes[0, 0].set_ylabel("Final Performance")
        axes[0, 0].set_title("Planning Steps vs Performance")
        axes[0, 0].grid(True, alpha=0.3)

    # Sample efficiency comparison
    sample_effs = [results[method].get("sample_efficiency", 0) for method in methods]
    axes[0, 1].bar(methods, sample_effs, alpha=0.7, color="lightgreen")
    axes[0, 1].set_title("Sample Efficiency")
    axes[0, 1].set_ylabel("Episodes to Target")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Learning efficiency comparison
    learning_effs = [
        results[method].get("learning_efficiency", 0) for method in methods
    ]
    axes[1, 0].bar(methods, learning_effs, alpha=0.7, color="lightblue")
    axes[1, 0].set_title("Learning Efficiency")
    axes[1, 0].set_ylabel("Average Reward")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Stability comparison
    stabilities = [results[method].get("stability", 0) for method in methods]
    axes[1, 1].bar(methods, stabilities, alpha=0.7, color="lightcoral")
    axes[1, 1].set_title("Learning Stability")
    axes[1, 1].set_ylabel("Stability Score")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_model_accuracy_analysis(
    results: Dict[str, Any], save_path: Optional[str] = None
) -> plt.Figure:
    """Plot model accuracy and uncertainty analysis"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Model Analysis", fontsize=16)

    methods = list(results.keys())

    # Model accuracy vs performance
    accuracies = []
    performances = []

    for method in methods:
        method_data = results[method]
        acc = method_data.get("model_accuracy", 0)
        perf = method_data.get("final_performance", 0)
        accuracies.append(acc)
        performances.append(perf)

    axes[0, 0].scatter(accuracies, performances, s=100, alpha=0.7)
    axes[0, 0].set_xlabel("Model Accuracy")
    axes[0, 0].set_ylabel("Final Performance")
    axes[0, 0].set_title("Model Accuracy vs Performance")
    axes[0, 0].grid(True, alpha=0.3)

    # Model uncertainty
    uncertainties = [results[method].get("model_uncertainty", 0) for method in methods]
    axes[0, 1].bar(methods, uncertainties, alpha=0.7, color="orange")
    axes[0, 1].set_title("Model Uncertainty")
    axes[0, 1].set_ylabel("Uncertainty")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Planning time analysis
    planning_times = [results[method].get("avg_planning_time", 0) for method in methods]
    axes[1, 0].bar(methods, planning_times, alpha=0.7, color="purple")
    axes[1, 0].set_title("Average Planning Time")
    axes[1, 0].set_ylabel("Time (seconds)")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Computational efficiency
    comp_times = [results[method].get("total_training_time", 0) for method in methods]
    axes[1, 1].bar(methods, comp_times, alpha=0.7, color="brown")
    axes[1, 1].set_title("Total Training Time")
    axes[1, 1].set_ylabel("Time (seconds)")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_summary_plots(
    results: Dict[str, Any], save_dir: str = "visualizations"
) -> None:
    """Create all summary plots and save to directory"""

    os.makedirs(save_dir, exist_ok=True)

    print("📊 Creating summary visualizations...")

    # Learning curves
    plot_learning_curves(
        results,
        save_path=os.path.join(save_dir, "learning_curves.png"),
        title="Model-Based RL Learning Curves",
    )

    # Performance comparison
    plot_comparison(
        results,
        metric="final_performance",
        save_path=os.path.join(save_dir, "performance_comparison.png"),
        title="Final Performance Comparison",
    )

    # Learning efficiency comparison
    plot_comparison(
        results,
        metric="learning_efficiency",
        save_path=os.path.join(save_dir, "learning_efficiency.png"),
        title="Learning Efficiency Comparison",
    )

    # Sample efficiency comparison
    plot_comparison(
        results,
        metric="sample_efficiency",
        save_path=os.path.join(save_dir, "sample_efficiency.png"),
        title="Sample Efficiency Comparison",
    )

    # Performance heatmap
    plot_performance_heatmap(
        results, save_path=os.path.join(save_dir, "performance_heatmap.png")
    )

    # Planning analysis
    plot_planning_analysis(
        results, save_path=os.path.join(save_dir, "planning_analysis.png")
    )

    # Model analysis
    plot_model_accuracy_analysis(
        results, save_path=os.path.join(save_dir, "model_analysis.png")
    )

    print(f"✅ All visualizations saved to {save_dir}/")


def demonstrate_visualization():
    """Demonstrate visualization functions"""
    print("Visualization Functions Demonstration")
    print("=" * 40)

    # Create sample results
    np.random.seed(42)
    sample_results = {
        "Q-Learning": {
            "final_performance": 0.75,
            "learning_efficiency": 0.65,
            "sample_efficiency": 80,
            "stability": 0.8,
            "episode_rewards": np.random.normal(0.7, 0.1, 100).tolist(),
            "episode_lengths": np.random.randint(15, 25, 100).tolist(),
        },
        "Dyna-Q (5)": {
            "final_performance": 0.85,
            "learning_efficiency": 0.78,
            "sample_efficiency": 60,
            "stability": 0.9,
            "planning_steps": 5,
            "episode_rewards": np.random.normal(0.8, 0.08, 100).tolist(),
            "episode_lengths": np.random.randint(12, 20, 100).tolist(),
        },
        "Dyna-Q (20)": {
            "final_performance": 0.92,
            "learning_efficiency": 0.85,
            "sample_efficiency": 45,
            "stability": 0.95,
            "planning_steps": 20,
            "episode_rewards": np.random.normal(0.9, 0.05, 100).tolist(),
            "episode_lengths": np.random.randint(10, 18, 100).tolist(),
        },
    }

    # Create visualizations
    create_summary_plots(sample_results, save_dir="demo_visualizations")

    print("✅ Visualization demonstration complete!")
    print("📁 Check the demo_visualizations/ folder for results")


if __name__ == "__main__":
    demonstrate_visualization()
