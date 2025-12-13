import torch
import numpy as np
import random
import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def get_device() -> torch.device:
    """Returns the appropriate device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_n_gpus() -> int:
    """Returns the number of available GPUs."""
    return torch.cuda.device_count()

def setup_logging(log_dir: str = "./logs") -> None:
    """Sets up a basic logging directory and configuration (can be extended)."""
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {os.path.abspath(log_dir)}")

def save_model(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    episode: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Saves the model state dictionary, and optionally optimizer, episode, and metrics."""
    state = {
        "model_state_dict": model.state_dict(),
        "episode": episode,
        "metrics": metrics,
    }
    if optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)
    print(f"Model saved to {path}")

def load_model(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Loads a model's state dictionary, and optionally optimizer state."""
    checkpoint = torch.load(path, map_location=get_device())
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Model loaded from {path}")
    return {
        "episode": checkpoint.get("episode"),
        "metrics": checkpoint.get("metrics"),
    }

def plot_learning_curves(
    rewards: Dict[str, List[float]],
    losses: Dict[str, List[float]],
    title: str = "Learning Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots reward and loss curves for multiple algorithms.

    Args:
        rewards (Dict[str, List[float]]): Dictionary of algorithm names to lists of episode rewards.
        losses (Dict[str, List[float]]): Dictionary of algorithm names to lists of training losses.
        title (str): Title of the plot.
        save_path (Optional[str]): Path to save the figure. If None, displays the figure.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    # Plot Rewards
    ax = axes[0]
    for algo_name, algo_rewards in rewards.items():
        ax.plot(algo_rewards, label=algo_name)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("Episode Rewards")
    ax.legend()
    ax.grid(True)

    # Plot Losses
    ax = axes[1]
    for algo_name, algo_losses in losses.items():
        if algo_losses: # Only plot if losses are recorded
            ax.plot(algo_losses, label=algo_name)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True)
    ax.set_yscale("log") # Often losses are better viewed on log scale

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show()
    return fig

def plot_multi_agent_performance(
    metrics: Dict[str, Dict[str, List[float]]],
    metric_name: str,
    title: str,
    ylabel: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots a specific multi-agent performance metric over time.

    Args:
        metrics (Dict[str, Dict[str, List[float]]]): Dictionary where keys are algorithm names,
                                                    values are dictionaries of metric names to lists of values.
        metric_name (str): The specific metric to plot (e.g., 'episode_rewards', 'success_rates').
        title (str): Title of the plot.
        ylabel (str): Label for the Y-axis.
        save_path (Optional[str]): Path to save the figure. If None, displays the figure.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo_name, algo_metrics in metrics.items():
        if metric_name in algo_metrics:
            ax.plot(algo_metrics[metric_name], label=algo_name)
    ax.set_xlabel("Episode / Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show()
    return fig

