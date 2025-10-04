import pickle
import json
import numpy as np
import os
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt


def save_model(agent, filename: str, save_dir: str = "models"):
    """
    Save trained agent model to disk

    Args:
        agent: Trained RL agent
        filename: Name of the file to save
        save_dir: Directory to save the model
    """
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, filename)

    # Extract model data
    model_data = {
        "class_name": agent.__class__.__name__,
        "Q": dict(agent.Q) if hasattr(agent, "Q") else {},
        "V": dict(agent.V) if hasattr(agent, "V") else {},
        "alpha": getattr(agent, "alpha", None),
        "gamma": getattr(agent, "gamma", None),
        "epsilon": getattr(agent, "epsilon", None),
        "episode_rewards": getattr(agent, "episode_rewards", []),
        "episode_steps": getattr(agent, "episode_steps", []),
        "epsilon_history": getattr(agent, "epsilon_history", []),
    }

    with open(filepath, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {filepath}")


def load_model(filename: str, save_dir: str = "models"):
    """
    Load trained agent model from disk

    Args:
        filename: Name of the file to load
        save_dir: Directory containing the model

    Returns:
        Dictionary with model data
    """
    filepath = os.path.join(save_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with open(filepath, "rb") as f:
        model_data = pickle.load(f)

    print(f"Model loaded from {filepath}")
    return model_data


def export_results(
    results: Dict[str, Any], filename: str, save_dir: str = "visualizations"
):
    """
    Export experiment results to various formats

    Args:
        results: Dictionary containing results
        filename: Base filename (without extension)
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Export as JSON
    json_path = os.path.join(save_dir, f"{filename}.json")
    with open(json_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        json_results[key][sub_key] = sub_value.tolist()
                    else:
                        json_results[key][sub_key] = sub_value
            else:
                json_results[key] = value

        json.dump(json_results, f, indent=2)

    # Export as CSV if results contain tabular data
    if "episode_rewards" in results:
        csv_path = os.path.join(save_dir, f"{filename}.csv")
        df = create_results_dataframe(results)
        df.to_csv(csv_path, index=False)

    print(f"Results exported to {save_dir}/{filename}.*")


def create_results_dataframe(results: Dict[str, Any]):
    """Create pandas DataFrame from results"""
    import pandas as pd

    data = {}

    if "episode_rewards" in results:
        data["episode"] = list(range(len(results["episode_rewards"])))
        data["reward"] = results["episode_rewards"]

    if "episode_steps" in results:
        data["steps"] = results["episode_steps"]

    if "epsilon_history" in results:
        data["epsilon"] = results["epsilon_history"]

    return pd.DataFrame(data)


def create_summary_report(
    results_dict: Dict[str, Any], save_dir: str = "visualizations"
):
    """
    Create a comprehensive summary report of all experiments

    Args:
        results_dict: Dictionary of {experiment_name: results}
        save_dir: Directory to save the report
    """
    os.makedirs(save_dir, exist_ok=True)

    report_path = os.path.join(save_dir, "experiment_summary_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TEMPORAL DIFFERENCE LEARNING - EXPERIMENT SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated on: {np.datetime64('now')}\n")
        f.write(f"Total experiments: {len(results_dict)}\n\n")

        for exp_name, results in results_dict.items():
            f.write(f"EXPERIMENT: {exp_name}\n")
            f.write("-" * 50 + "\n")

            if "avg_reward" in results:
                f.write(f"Average Reward: {results['avg_reward']:.3f}\n")
            if "success_rate" in results:
                f.write(f"Success Rate: {results['success_rate']*100:.1f}%\n")
            if "avg_steps" in results:
                f.write(f"Average Steps: {results['avg_steps']:.1f}\n")
            if "total_episodes" in results:
                f.write(f"Total Episodes: {results['total_episodes']}\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Summary report saved to {report_path}")


def visualize_model_comparison(
    models_dict: Dict[str, Any], save_dir: str = "visualizations"
):
    """
    Create visualization comparing multiple models

    Args:
        models_dict: Dictionary of {model_name: model_data}
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Learning curves comparison
    ax1 = axes[0, 0]
    for model_name, model_data in models_dict.items():
        if "episode_rewards" in model_data and model_data["episode_rewards"]:
            rewards = model_data["episode_rewards"]
            ax1.plot(rewards, alpha=0.7, label=model_name, linewidth=1)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Learning Curves Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Final performance comparison
    ax2 = axes[0, 1]
    model_names = []
    final_performances = []

    for model_name, model_data in models_dict.items():
        if "episode_rewards" in model_data and model_data["episode_rewards"]:
            rewards = model_data["episode_rewards"]
            final_perf = (
                np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            )
            model_names.append(model_name)
            final_performances.append(final_perf)

    if model_names:
        bars = ax2.bar(model_names, final_performances, alpha=0.7)
        ax2.set_ylabel("Final Performance")
        ax2.set_title("Final Performance Comparison")
        ax2.tick_params(axis="x", rotation=45)

    # Episode steps comparison
    ax3 = axes[1, 0]
    for model_name, model_data in models_dict.items():
        if "episode_steps" in model_data and model_data["episode_steps"]:
            steps = model_data["episode_steps"]
            ax3.plot(steps, alpha=0.7, label=model_name, linewidth=1)

    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Steps to Goal")
    ax3.set_title("Steps per Episode Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Exploration rate comparison
    ax4 = axes[1, 1]
    for model_name, model_data in models_dict.items():
        if "epsilon_history" in model_data and model_data["epsilon_history"]:
            epsilon_history = model_data["epsilon_history"]
            ax4.plot(epsilon_history, alpha=0.7, label=model_name, linewidth=2)

    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon (ε)")
    ax4.set_title("Exploration Rate Decay")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "model_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def backup_models(models_dir: str = "models", backup_dir: str = "model_backups"):
    """
    Create backup of all trained models

    Args:
        models_dir: Directory containing models
        backup_dir: Directory to store backups
    """
    import shutil
    from datetime import datetime

    if not os.path.exists(models_dir):
        print(f"No models directory found: {models_dir}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"backup_{timestamp}")

    os.makedirs(backup_path, exist_ok=True)
    shutil.copytree(models_dir, backup_path, dirs_exist_ok=True)

    print(f"Models backed up to {backup_path}")


def cleanup_old_files(directory: str, file_pattern: str = "*.png", max_files: int = 10):
    """
    Clean up old visualization files to save space

    Args:
        directory: Directory to clean
        file_pattern: Pattern of files to clean
        max_files: Maximum number of files to keep
    """
    import glob

    files = glob.glob(os.path.join(directory, file_pattern))

    if len(files) <= max_files:
        return

    # Sort by modification time (oldest first)
    files.sort(key=os.path.getmtime)

    # Remove oldest files
    files_to_remove = files[:-max_files]
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"Removed old file: {file_path}")
        except OSError as e:
            print(f"Error removing {file_path}: {e}")

    print(f"Cleaned up {len(files_to_remove)} old files from {directory}")

