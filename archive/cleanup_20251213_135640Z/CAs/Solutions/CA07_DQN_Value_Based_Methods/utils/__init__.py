"""
Utility functions for CA07 DQN experiments
==========================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os
import time
from collections import defaultdict, deque
import warnings

warnings.filterwarnings("ignore")


# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def smooth_curve(data: List[float], window: int = 20) -> np.ndarray:
    """Smooth a curve using moving average"""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a dataset"""
    data_array = np.array(data)
    return {
        "mean": np.mean(data_array),
        "std": np.std(data_array),
        "min": np.min(data_array),
        "max": np.max(data_array),
        "median": np.median(data_array),
        "q25": np.percentile(data_array, 25),
        "q75": np.percentile(data_array, 75),
    }


def find_convergence_episode(
    scores: List[float], target: float = 180, window: int = 20
) -> int:
    """Find episode where agent converges to target score"""
    smoothed = np.convolve(scores, np.ones(window) / window, mode="valid")
    converged_idx = np.where(smoothed >= target)[0]
    return converged_idx[0] if len(converged_idx) > 0 else len(smoothed)


def calculate_learning_efficiency(scores: List[float], target: float = 180) -> float:
    """Calculate learning efficiency (area under curve)"""
    converged_episode = find_convergence_episode(scores, target)
    if converged_episode >= len(scores):
        return 0.0

    # Calculate area under curve up to convergence
    area = np.trapz(scores[:converged_episode])
    max_possible_area = target * converged_episode
    return area / max_possible_area if max_possible_area > 0 else 0.0


class PerformanceTracker:
    """Track performance metrics during training"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.scores = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.epsilon_values = deque(maxlen=window_size)

    def update(
        self,
        score: float,
        loss: float = 0.0,
        episode_length: int = 0,
        epsilon: float = 0.0,
    ):
        """Update tracking metrics"""
        self.scores.append(score)
        self.losses.append(loss)
        self.episode_lengths.append(episode_length)
        self.epsilon_values.append(epsilon)

    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics"""
        return {
            "mean_score": np.mean(self.scores) if self.scores else 0.0,
            "std_score": np.std(self.scores) if self.scores else 0.0,
            "mean_loss": np.mean(self.losses) if self.losses else 0.0,
            "mean_length": (
                np.mean(self.episode_lengths) if self.episode_lengths else 0.0
            ),
            "current_epsilon": self.epsilon_values[-1] if self.epsilon_values else 0.0,
        }

    def is_converged(self, target: float = 180, min_episodes: int = 50) -> bool:
        """Check if agent has converged"""
        if len(self.scores) < min_episodes:
            return False
        return np.mean(list(self.scores)[-min_episodes:]) >= target


class ExperimentLogger:
    """Log experiment results and metrics"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logs = defaultdict(list)

    def log(self, key: str, value: Any, episode: int = None):
        """Log a value"""
        if episode is not None:
            self.logs[key].append((episode, value))
        else:
            self.logs[key].append(value)

    def log_episode(
        self,
        episode: int,
        score: float,
        loss: float = 0.0,
        episode_length: int = 0,
        epsilon: float = 0.0,
    ):
        """Log episode results"""
        self.log("episode", episode)
        self.log("score", score)
        self.log("loss", loss)
        self.log("episode_length", episode_length)
        self.log("epsilon", epsilon)

    def save_logs(self, filename: str = "experiment_log.json"):
        """Save logs to file"""
        # Convert to serializable format
        serializable_logs = {}
        for key, values in self.logs.items():
            if isinstance(values[0], tuple):
                serializable_logs[key] = [(ep, float(val)) for ep, val in values]
            else:
                serializable_logs[key] = [float(val) for val in values]

        with open(os.path.join(self.log_dir, filename), "w") as f:
            json.dump(serializable_logs, f, indent=2)

    def load_logs(self, filename: str = "experiment_log.json"):
        """Load logs from file"""
        with open(os.path.join(self.log_dir, filename), "r") as f:
            self.logs = json.load(f)


def create_summary_plot(results: Dict[str, Dict], save_path: str = None):
    """Create a comprehensive summary plot"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    agent_names = list(results.keys())
    colors = ["blue", "green", "red", "purple", "orange", "brown"]

    # Learning curves
    ax1 = axes[0, 0]
    for i, (name, result) in enumerate(results.items()):
        scores = result["scores"]
        smoothed = smooth_curve(scores, window=20)
        ax1.plot(smoothed, label=name, color=colors[i % len(colors)], linewidth=2)
    ax1.set_title("Learning Curves")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Smoothed Score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Final performance
    ax2 = axes[0, 1]
    final_scores = [np.mean(result["scores"][-50:]) for result in results.values()]
    bars = ax2.bar(
        agent_names, final_scores, alpha=0.7, color=colors[: len(agent_names)]
    )
    ax2.set_title("Final Performance")
    ax2.set_ylabel("Final Average Score")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, score in zip(bars, final_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{score:.1f}",
            ha="center",
            va="bottom",
        )

    # Training stability
    ax3 = axes[0, 2]
    stability_scores = [np.std(result["scores"][-100:]) for result in results.values()]
    bars = ax3.bar(agent_names, stability_scores, alpha=0.7, color="orange")
    ax3.set_title("Training Stability")
    ax3.set_ylabel("Score Standard Deviation")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)

    # Loss curves
    ax4 = axes[1, 0]
    for i, (name, result) in enumerate(results.items()):
        if "losses" in result and result["losses"]:
            losses = result["losses"]
            smoothed_losses = smooth_curve(losses, window=20)
            ax4.plot(
                smoothed_losses, label=name, color=colors[i % len(colors)], linewidth=2
            )
    ax4.set_title("Loss Curves")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Smoothed Loss")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Convergence analysis
    ax5 = axes[1, 1]
    convergence_episodes = []
    for result in results.values():
        conv_ep = find_convergence_episode(result["scores"])
        convergence_episodes.append(conv_ep)

    bars = ax5.bar(agent_names, convergence_episodes, alpha=0.7, color="red")
    ax5.set_title("Convergence Episodes")
    ax5.set_ylabel("Episode")
    ax5.tick_params(axis="x", rotation=45)
    ax5.grid(True, alpha=0.3)

    # Performance summary
    ax6 = axes[1, 2]
    metrics = ["Final Score", "Stability", "Convergence"]
    best_agent_idx = np.argmax(final_scores)

    # Normalize metrics for comparison
    normalized_final = np.array(final_scores) / np.max(final_scores)
    normalized_stability = 1 - np.array(stability_scores) / np.max(
        stability_scores
    )  # Lower is better
    normalized_convergence = 1 - np.array(convergence_episodes) / np.max(
        convergence_episodes
    )  # Lower is better

    x = np.arange(len(metrics))
    width = 0.25

    ax6.bar(x - width, normalized_final, width, label="Final Score", alpha=0.7)
    ax6.bar(x, normalized_stability, width, label="Stability", alpha=0.7)
    ax6.bar(x + width, normalized_convergence, width, label="Convergence", alpha=0.7)

    ax6.set_title("Normalized Performance Metrics")
    ax6.set_ylabel("Normalized Score")
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def benchmark_agent(
    agent_class, env_name: str, num_runs: int = 5, episodes: int = 200, **agent_kwargs
) -> Dict[str, Any]:
    """Benchmark agent across multiple runs"""
    results = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        set_seed(42 + run)

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = agent_class(state_dim=state_dim, action_dim=action_dim, **agent_kwargs)

        run_rewards = []
        for episode in range(episodes):
            reward, _ = agent.train_episode(env, max_steps=500)
            run_rewards.append(reward)

        results.append(run_rewards)
        env.close()

    # Calculate statistics across runs
    results_array = np.array(results)

    return {
        "mean_scores": np.mean(results_array, axis=0),
        "std_scores": np.std(results_array, axis=0),
        "final_mean": np.mean(results_array[:, -50:]),
        "final_std": np.std(results_array[:, -50:]),
        "individual_runs": results,
        "convergence_episode": find_convergence_episode(np.mean(results_array, axis=0)),
    }


def save_results(results: Dict[str, Any], filename: str = "results.json"):
    """Save results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    serializable_results[key][sub_key] = sub_value.tolist()
                else:
                    serializable_results[key][sub_key] = sub_value
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value

    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)


def load_results(filename: str = "results.json") -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(filename, "r") as f:
        return json.load(f)


def print_experiment_summary(results: Dict[str, Dict]):
    """Print a summary of experiment results"""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for agent_name, result in results.items():
        scores = result["scores"]
        stats = calculate_statistics(scores)
        conv_ep = find_convergence_episode(scores)
        efficiency = calculate_learning_efficiency(scores)

        print(f"\n{agent_name}:")
        print(
            f"  Final Score (last 50): {np.mean(scores[-50:]):.2f} ± {np.std(scores[-50:]):.2f}"
        )
        print(f"  Best Score: {stats['max']:.2f}")
        print(f"  Convergence Episode: {conv_ep}")
        print(f"  Learning Efficiency: {efficiency:.3f}")
        print(f"  Training Stability: {np.std(scores[-100:]):.2f}")

    # Find best agent
    best_agent = max(results.keys(), key=lambda x: np.mean(results[x]["scores"][-50:]))
    print(f"\nBest Agent: {best_agent}")
    print("=" * 60)


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a progress bar string"""
    progress = current / total
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {progress:.1%} ({current}/{total})"


def format_time(seconds: float) -> str:
    """Format time in a readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class Timer:
    """Simple timer context manager"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"{self.name} completed in {format_time(elapsed)}")


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and information"""
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        ),
        "device_name": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else None
        ),
    }
    return gpu_info


def get_memory_usage() -> Dict[str, float]:
    """Get memory usage information"""
    memory_info = {}

    if torch.cuda.is_available():
        memory_info["gpu_allocated"] = torch.cuda.memory_allocated() / (1024**3)  # GB
        memory_info["gpu_cached"] = torch.cuda.memory_reserved() / (1024**3)  # GB

    import psutil

    process = psutil.Process()
    memory_info["cpu_memory"] = process.memory_info().rss / (1024**3)  # GB

    return memory_info


