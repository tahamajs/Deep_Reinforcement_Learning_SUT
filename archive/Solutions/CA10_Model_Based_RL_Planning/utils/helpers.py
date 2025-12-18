"""
Helper functions for Model-Based RL
"""

import numpy as np
import torch
import random
import os
import pickle
import json
from typing import Any, Dict, List, Optional, Union
import time
import sys


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_directories(base_path: str = ".", dirs: List[str] = None) -> None:
    """Create necessary directories"""
    if dirs is None:
        dirs = ["visualizations", "logs", "results", "models", "data"]

    for dir_name in dirs:
        dir_path = os.path.join(base_path, dir_name)
        os.makedirs(dir_path, exist_ok=True)


def save_results(
    results: Dict[str, Any], filepath: str, format: str = "pickle"
) -> None:
    """Save results to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if format == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
    elif format == "json":
        # Convert numpy arrays to lists for JSON serialization
        json_results = convert_for_json(results)
        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2)
    elif format == "npy":
        np.save(filepath, results)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Results saved to {filepath}")


def load_results(filepath: str, format: str = "pickle") -> Dict[str, Any]:
    """Load results from file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if format == "pickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif format == "json":
        with open(filepath, "r") as f:
            return json.load(f)
    elif format == "npy":
        return np.load(filepath, allow_pickle=True).item()
    else:
        raise ValueError(f"Unsupported format: {format}")


def convert_for_json(obj: Any) -> Any:
    """Convert numpy arrays and other non-JSON-serializable objects"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_for_json(item) for item in obj]
    else:
        return obj


def smooth_curve(data: List[float], window_size: int = 10) -> List[float]:
    """Smooth a curve using moving average"""
    if len(data) < window_size:
        return data

    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        smoothed.append(np.mean(data[start_idx : i + 1]))

    return smoothed


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of numbers"""
    if not data:
        return {}

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


def time_function(func):
    """Decorator to time function execution"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def print_progress(
    current: int,
    total: int,
    prefix: str = "Progress",
    suffix: str = "",
    decimals: int = 1,
    length: int = 50,
) -> None:
    """Print progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")

    if current == total:
        print()


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def log_to_file(message: str, filepath: str = "logs/execution.log") -> None:
    """Log message to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


class Timer:
    """Context manager for timing code blocks"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"{self.name} completed in {format_time(elapsed)}")


def validate_environment(env) -> bool:
    """Validate environment has required methods"""
    required_methods = ["reset", "step", "num_states", "num_actions"]

    for method in required_methods:
        if not hasattr(env, method):
            print(f"Environment missing required method: {method}")
            return False

    return True


def validate_agent(agent) -> bool:
    """Validate agent has required methods"""
    required_methods = ["select_action", "train_episode"]

    for method in required_methods:
        if not hasattr(agent, method):
            print(f"Agent missing required method: {method}")
            return False

    return True


def create_config_dict(**kwargs) -> Dict[str, Any]:
    """Create configuration dictionary with default values"""
    config = {
        "seed": 42,
        "num_episodes": 100,
        "max_steps": 200,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 0.1,
        "batch_size": 32,
        "hidden_dim": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    config.update(kwargs)
    return config


def print_config(config: Dict[str, Any]) -> None:
    """Print configuration in a nice format"""
    print("🔧 Configuration:")
    print("=" * 30)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()


def demonstrate_helpers():
    """Demonstrate helper functions"""
    print("Helper Functions Demonstration")
    print("=" * 40)

    # Set seed
    set_seed(42)
    print("✅ Random seed set to 42")

    # Create directories
    create_directories()
    print("✅ Directories created")

    # Calculate statistics
    sample_data = np.random.normal(0, 1, 100)
    stats = calculate_statistics(sample_data)
    print(f"✅ Statistics calculated: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    # Smooth curve
    noisy_data = np.sin(np.linspace(0, 2 * np.pi, 50)) + np.random.normal(0, 0.1, 50)
    smoothed = smooth_curve(noisy_data.tolist())
    print(f"✅ Curve smoothed: {len(smoothed)} points")

    # Timer
    with Timer("Sample operation"):
        time.sleep(0.1)

    # Progress bar
    print("✅ Progress bar demonstration:")
    for i in range(11):
        print_progress(i, 10, prefix="Testing")
        time.sleep(0.1)
    print()

    # Memory usage
    memory = get_memory_usage()
    print(f"✅ Current memory usage: {memory:.2f} MB")

    print("\n🎉 Helper functions demonstration complete!")


if __name__ == "__main__":
    demonstrate_helpers()
