"""
Utilities for Advanced Deep RL

This module contains utility functions and classes for advanced deep RL experiments,
including visualization tools, data processing, and common RL utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import random
import time
import os
from collections import defaultdict, deque
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Experience replay buffer for RL agents."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.epsilon = 1e-6

    def push(self, state, action, reward, next_state, done):
        """Store a transition with maximum priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample a batch with priorities."""
        if len(self.buffer) == 0:
            return None

        priorities = self.priorities[: len(self.buffer)]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device),
            torch.FloatTensor(weights).to(device),
            indices,
        )

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self):
        return len(self.buffer)


class RunningStats:
    """Running statistics calculator."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        """Update statistics with new value."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    def get_std(self):
        return np.sqrt(self.get_variance())


class Logger:
    """Simple logging utility for experiments."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = defaultdict(list)
        self.start_time = time.time()

    def log(self, key: str, value: float):
        """Log a metric."""
        self.metrics[key].append(value)

    def log_dict(self, metrics: Dict[str, float]):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log(key, value)

    def save(self, filename: str = "experiment_log.npy"):
        """Save logged metrics."""
        filepath = os.path.join(self.log_dir, filename)
        np.save(filepath, dict(self.metrics))
        print(f"📊 Logs saved to {filepath}")

    def plot(self, keys: List[str] = None, save_path: str = None):
        """Plot logged metrics."""
        if keys is None:
            keys = list(self.metrics.keys())

        fig, axes = plt.subplots(len(keys), 1, figsize=(10, 4 * len(keys)))
        if len(keys) == 1:
            axes = [axes]

        for i, key in enumerate(keys):
            if key in self.metrics:
                values = self.metrics[key]
                axes[i].plot(values, label=key)
                axes[i].set_title(f"{key} over time")
                axes[i].set_xlabel("Step")
                axes[i].set_ylabel(key)
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


class NeuralNetworkUtils:
    """Utility functions for neural networks."""

    @staticmethod
    def init_weights(module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def count_parameters(model):
        """Count trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def soft_update(target_model, source_model, tau: float):
        """Soft update target network parameters."""
        for target_param, source_param in zip(
            target_model.parameters(), source_model.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )

    @staticmethod
    def hard_update(target_model, source_model):
        """Hard update target network parameters."""
        target_model.load_state_dict(source_model.state_dict())


class VisualizationUtils:
    """Visualization utilities for RL experiments."""

    @staticmethod
    def plot_learning_curve(
        rewards: List[float],
        window: int = 100,
        title: str = "Learning Curve",
        save_path: str = None,
    ):
        """Plot learning curve with moving average."""
        plt.figure(figsize=(10, 6))

        if len(rewards) > window:
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            plt.plot(
                moving_avg, label=f"Moving Average ({window} episodes)", linewidth=2
            )

        plt.plot(rewards, alpha=0.6, label="Raw Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_multiple_curves(
        curves: Dict[str, List[float]],
        window: int = 100,
        title: str = "Comparison",
        save_path: str = None,
    ):
        """Plot multiple learning curves."""
        plt.figure(figsize=(12, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))

        for i, (label, rewards) in enumerate(curves.items()):
            if len(rewards) > window:
                moving_avg = pd.Series(rewards).rolling(window=window).mean()
                plt.plot(
                    moving_avg, label=f"{label} (MA)", linewidth=2, color=colors[i]
                )

            plt.plot(rewards, alpha=0.4, color=colors[i])

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_value_function(
        values: np.ndarray, title: str = "Value Function", save_path: str = None
    ):
        """Plot value function as heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(values, cmap="viridis", annot=True, fmt=".2f")
        plt.title(title)
        plt.xlabel("State/Action")
        plt.ylabel("State/Action")

        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_policy(
        policy: np.ndarray,
        title: str = "Policy",
        action_names: List[str] = None,
        save_path: str = None,
    ):
        """Plot policy as heatmap."""
        plt.figure(figsize=(10, 8))

        if action_names:
            sns.heatmap(
                policy,
                cmap="Blues",
                annot=True,
                fmt=".2f",
                xticklabels=action_names,
                yticklabels=range(len(policy)),
            )
        else:
            sns.heatmap(policy, cmap="Blues", annot=True, fmt=".2f")

        plt.title(title)
        plt.xlabel("Action")
        plt.ylabel("State")

        if save_path:
            plt.savefig(save_path)
        plt.show()


class EnvironmentUtils:
    """Utilities for working with RL environments."""

    @staticmethod
    def normalize_state(
        state: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None
    ) -> np.ndarray:
        """Normalize state observations."""
        if mean is None or std is None:
            return state
        return (state - mean) / (std + 1e-8)

    @staticmethod
    def preprocess_state(state: np.ndarray, flatten: bool = True) -> torch.Tensor:
        """Preprocess state for neural network input."""
        if flatten:
            state = state.flatten()
        return torch.FloatTensor(state).unsqueeze(0).to(device)

    @staticmethod
    def evaluate_agent(
        agent, env, num_episodes: int = 10, render: bool = False
    ) -> Dict[str, float]:
        """Evaluate an agent on an environment."""
        rewards = []
        lengths = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                if render:
                    env.render()

                action = agent.get_action(state, epsilon=0.0)  # Greedy policy
                next_state, reward, done, info = env.step(action)

                episode_reward += reward
                episode_length += 1
                state = next_state

                if episode_length > 1000:  # Safety limit
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "rewards": rewards,
            "lengths": lengths,
        }


class ExperimentUtils:
    """Utilities for running RL experiments."""

    @staticmethod
    def set_random_seeds(seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def create_experiment_grid(**kwargs) -> List[Dict[str, Any]]:
        """Create parameter grid for experiments."""
        import itertools

        keys = kwargs.keys()
        values = kwargs.values()
        combinations = itertools.product(*values)

        return [dict(zip(keys, combination)) for combination in combinations]

    @staticmethod
    def run_parameter_sweep(
        agent_class,
        env,
        param_grid: List[Dict[str, Any]],
        num_episodes: int = 100,
        num_seeds: int = 3,
    ) -> pd.DataFrame:
        """Run parameter sweep experiment."""
        results = []

        for params in param_grid:
            print(f"Testing parameters: {params}")

            param_results = []
            for seed in range(num_seeds):
                ExperimentUtils.set_random_seeds(seed)

                agent = agent_class(**params)

                rewards = []
                for episode in range(num_episodes):
                    state = env.reset()
                    episode_reward = 0
                    done = False

                    while not done:
                        action = agent.get_action(state)
                        next_state, reward, done, info = env.step(action)

                        agent.store_experience(state, action, reward, next_state, done)
                        agent.update()

                        episode_reward += reward
                        state = next_state

                    rewards.append(episode_reward)

                param_results.append(np.mean(rewards[-20:]))  # Final performance

            results.append(
                {
                    **params,
                    "mean_final_reward": np.mean(param_results),
                    "std_final_reward": np.std(param_results),
                }
            )

        return pd.DataFrame(results)


def set_device(gpu_id: int = None):
    """Set the device for PyTorch computations."""
    global device
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def get_device():
    """Get current device."""
    return device


def to_tensor(data, dtype=torch.float32):
    """Convert data to tensor on current device."""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype).to(device)
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data, dtype=dtype).to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(dtype).to(device)
    else:
        return torch.tensor(data, dtype=dtype).to(device)
