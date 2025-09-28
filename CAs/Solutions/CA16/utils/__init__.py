"""
Utilities Module

This module contains common utilities used across all CA16 modules:
- Data processing and trajectory handling
- Evaluation and metrics
- Visualization functions
- Common RL utilities
- Configuration management
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from collections import defaultdict, deque
import pickle
import os
import json
import time
from datetime import datetime
import logging
import warnings
from pathlib import Path


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


class TrajectoryBuffer:
    """
    Buffer for storing and managing trajectory data.

    Provides efficient storage and sampling of trajectories for training.
    """

    def __init__(self, max_size: int = 10000, device: str = 'cpu'):
        self.max_size = max_size
        self.device = device

        # Storage
        self.trajectories: List[Dict[str, Any]] = []
        self.current_trajectory: Dict[str, Any] = self._new_trajectory()

        # Statistics
        self.total_transitions = 0
        self.episode_count = 0

    def _new_trajectory(self) -> Dict[str, Any]:
        """Create a new empty trajectory."""
        return {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'infos': []
        }

    def add_transition(self, state: np.ndarray, action: np.ndarray,
                      reward: float, next_state: np.ndarray,
                      done: bool, info: Optional[Dict] = None):
        """
        Add a transition to the current trajectory.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            info: Additional information
        """
        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['next_states'].append(next_state)
        self.current_trajectory['dones'].append(done)
        self.current_trajectory['infos'].append(info or {})

        self.total_transitions += 1

        if done:
            self._finish_trajectory()

    def _finish_trajectory(self):
        """Finish the current trajectory and store it."""
        if self.current_trajectory['states']:  # Only if not empty
            # Convert lists to numpy arrays for efficiency
            trajectory = {
                'states': np.array(self.current_trajectory['states']),
                'actions': np.array(self.current_trajectory['actions']),
                'rewards': np.array(self.current_trajectory['rewards']),
                'next_states': np.array(self.current_trajectory['next_states']),
                'dones': np.array(self.current_trajectory['dones']),
                'infos': self.current_trajectory['infos'],
                'length': len(self.current_trajectory['states']),
                'return': np.sum(self.current_trajectory['rewards'])
            }

            self.trajectories.append(trajectory)
            self.episode_count += 1

            # Maintain buffer size
            if len(self.trajectories) > self.max_size:
                self.trajectories.pop(0)

        # Start new trajectory
        self.current_trajectory = self._new_trajectory()

    def sample_trajectory(self) -> Optional[Dict[str, Any]]:
        """Sample a random trajectory from the buffer."""
        if not self.trajectories:
            return None
        return np.random.choice(self.trajectories)

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of trajectories."""
        if len(self.trajectories) < batch_size:
            return self.trajectories.copy()

        indices = np.random.choice(len(self.trajectories), batch_size, replace=False)
        return [self.trajectories[i] for i in indices]

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self.trajectories:
            return {'empty': True}

        lengths = [t['length'] for t in self.trajectories]
        returns = [t['return'] for t in self.trajectories]

        return {
            'num_trajectories': len(self.trajectories),
            'total_transitions': self.total_transitions,
            'avg_length': np.mean(lengths),
            'max_length': np.max(lengths),
            'min_length': np.min(lengths),
            'avg_return': np.mean(returns),
            'max_return': np.max(returns),
            'min_return': np.min(returns)
        }

    def save_to_disk(self, filepath: str):
        """Save trajectories to disk."""
        data = {
            'trajectories': self.trajectories,
            'statistics': self.get_statistics(),
            'timestamp': time.time()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_from_disk(self, filepath: str):
        """Load trajectories from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.trajectories = data['trajectories']
        self.total_transitions = sum(len(t['states']) for t in self.trajectories)
        self.episode_count = len(self.trajectories)


class MetricsTracker:
    """
    Comprehensive metrics tracking for RL experiments.

    Tracks various metrics during training and evaluation.
    """

    def __init__(self, save_dir: Optional[str] = None):
        self.metrics: Dict[str, List[Any]] = defaultdict(list)
        self.start_time = time.time()
        self.save_dir = save_dir

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def log_metric(self, name: str, value: Any, step: Optional[int] = None):
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Training step (optional)
        """
        entry = {
            'value': value,
            'timestamp': time.time(),
            'step': step or len(self.metrics[name])
        }

        self.metrics[name].append(entry)

    def log_metrics(self, metrics_dict: Dict[str, Any], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)

    def get_metric_history(self, name: str) -> List[Dict[str, Any]]:
        """Get history of a specific metric."""
        return self.metrics[name]

    def get_latest_metric(self, name: str) -> Optional[Any]:
        """Get the latest value of a metric."""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]['value']
        return None

    def get_metric_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return {}

        values = [entry['value'] for entry in self.metrics[name]]

        # Handle different value types
        if isinstance(values[0], (int, float)):
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'latest': values[-1]
            }
        else:
            return {
                'count': len(values),
                'latest': values[-1]
            }

    def plot_metric(self, name: str, save_path: Optional[str] = None):
        """Plot a metric over time."""
        if name not in self.metrics:
            logger.warning(f"Metric {name} not found")
            return

        history = self.metrics[name]
        if not history:
            return

        # Extract data
        steps = [entry.get('step', i) for i, entry in enumerate(history)]
        values = [entry['value'] for entry in history]

        # Only plot numeric values
        if not isinstance(values[0], (int, float)):
            logger.warning(f"Cannot plot non-numeric metric {name}")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, 'b-', alpha=0.7)
        plt.title(f'{name.replace("_", " ").title()}')
        plt.xlabel('Step')
        plt.ylabel(name.replace("_", " ").title())
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_metrics(self, filepath: Optional[str] = None):
        """Save metrics to disk."""
        if not filepath and self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_dir, f'metrics_{timestamp}.json')

        if filepath:
            data = {
                'metrics': dict(self.metrics),
                'start_time': self.start_time,
                'end_time': time.time()
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

    def load_metrics(self, filepath: str):
        """Load metrics from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.metrics = defaultdict(list, data['metrics'])
        self.start_time = data['start_time']


class ExperimentConfig:
    """
    Configuration management for RL experiments.

    Provides a structured way to define and manage experiment parameters.
    """

    def __init__(self, **kwargs):
        # Default configurations
        self.config = {
            # Environment
            'env_name': 'CartPole-v1',
            'max_episode_steps': 1000,

            # Agent
            'agent_type': 'dqn',
            'learning_rate': 1e-3,
            'batch_size': 64,
            'gamma': 0.99,

            # Training
            'num_episodes': 1000,
            'max_steps_per_episode': 1000,
            'eval_freq': 100,
            'save_freq': 500,

            # Neural network
            'hidden_dims': [128, 128],
            'activation': 'relu',

            # Exploration
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,

            # Device
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',

            # Logging
            'log_dir': './logs',
            'save_dir': './checkpoints',
            'experiment_name': f'experiment_{int(time.time())}'
        }

        # Update with provided kwargs
        self.config.update(kwargs)

        # Create directories
        os.makedirs(self.config['log_dir'], exist_ok=True)
        os.makedirs(self.config['save_dir'], exist_ok=True)

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.config

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        self.config.update(updates)

    def save_config(self, filepath: Optional[str] = None) -> str:
        """Save configuration to file."""
        if not filepath:
            filepath = os.path.join(self.config['log_dir'], 'config.json')

        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)

        return filepath

    def load_config(self, filepath: str):
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)

        self.config.update(loaded_config)

    def print_config(self):
        """Print current configuration."""
        print("Experiment Configuration:")
        print("=" * 50)
        for key, value in sorted(self.config.items()):
            print(f"{key}: {value}")
        print("=" * 50)


def compute_returns_to_go(rewards: List[float], gamma: float = 0.99,
                         dones: Optional[List[bool]] = None) -> np.ndarray:
    """
    Compute returns-to-go for a trajectory.

    Args:
        rewards: List of rewards
        gamma: Discount factor
        dones: List of done flags (optional)

    Returns:
        Array of returns-to-go
    """
    if dones is None:
        dones = [False] * len(rewards)

    returns = []
    running_return = 0

    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            running_return = 0
        running_return = reward + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    return np.array(returns)


def normalize_observations(observations: np.ndarray,
                          epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize observations using running statistics.

    Args:
        observations: Array of observations
        epsilon: Small value for numerical stability

    Returns:
        Normalized observations, mean, std
    """
    mean = np.mean(observations, axis=0)
    std = np.std(observations, axis=0) + epsilon

    normalized = (observations - mean) / std

    return normalized, mean, std


def create_mlp_network(input_dim: int, output_dim: int,
                      hidden_dims: List[int] = [128, 128],
                      activation: str = 'relu',
                      output_activation: Optional[str] = None) -> nn.Module:
    """
    Create a multi-layer perceptron network.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: Hidden layer dimensions
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer

    Returns:
        MLP network
    """
    layers = []

    # Input layer
    layers.extend([
        nn.Linear(input_dim, hidden_dims[0]),
        get_activation(activation)
    ])

    # Hidden layers
    for i in range(len(hidden_dims) - 1):
        layers.extend([
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
            get_activation(activation)
        ])

    # Output layer
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if output_activation:
        layers.append(get_activation(output_activation))

    return nn.Sequential(*layers)


def get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.

    Args:
        name: Activation function name

    Returns:
        PyTorch activation module
    """
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU()
    }

    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")

    return activations[name]


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, filepath: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'timestamp': checkpoint.get('timestamp', 0)
    }


def plot_training_progress(metrics: Dict[str, List[float]],
                          save_path: Optional[str] = None):
    """
    Plot training progress metrics.

    Args:
        metrics: Dictionary of metric histories
        save_path: Path to save plot (optional)
    """
    num_metrics = len(metrics)
    if num_metrics == 0:
        return

    fig, axes = plt.subplots((num_metrics + 2) // 3, min(3, num_metrics),
                           figsize=(15, 5 * ((num_metrics + 2) // 3)))

    if num_metrics == 1:
        axes = [axes]

    axes = axes.flatten() if num_metrics > 1 else [axes]

    for i, (name, values) in enumerate(metrics.items()):
        if i >= len(axes):
            break

        ax = axes[i]
        ax.plot(values, 'b-', alpha=0.7)
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('Step')
        ax.set_ylabel(name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def evaluate_policy(env_fn: Callable, policy: Callable,
                   num_episodes: int = 10, max_steps: int = 1000,
                   render: bool = False) -> Dict[str, Any]:
    """
    Evaluate a policy on an environment.

    Args:
        env_fn: Function that returns environment
        policy: Policy function (state -> action)
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render environment

    Returns:
        Evaluation results
    """
    returns = []
    lengths = []

    for episode in range(num_episodes):
        env = env_fn()
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            if render:
                env.render()

            action = policy(state)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1
            state = next_state

        returns.append(episode_return)
        lengths.append(episode_length)

    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'returns': returns,
        'lengths': lengths
    }


def compute_discounted_sum(values: List[float], gamma: float = 0.99) -> float:
    """
    Compute discounted sum of values.

    Args:
        values: List of values
        gamma: Discount factor

    Returns:
        Discounted sum
    """
    discounted_sum = 0.0
    for i, value in enumerate(reversed(values)):
        discounted_sum = value + gamma * discounted_sum

    return discounted_sum


def safe_mean(values: List[float]) -> float:
    """Compute mean safely handling empty lists."""
    return np.mean(values) if values else 0.0


def safe_std(values: List[float]) -> float:
    """Compute standard deviation safely handling empty lists."""
    return np.std(values) if values else 0.0


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'error': 'psutil not available'}


def get_gpu_memory_usage() -> Dict[str, Any]:
    """Get GPU memory usage if available."""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_memory_free': gpu.memoryFree,
                'gpu_memory_util': gpu.memoryUtil * 100
            }
    except ImportError:
        pass

    return {'error': 'GPU monitoring not available'}