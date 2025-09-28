"""
Utilities Module

This module contains utility functions and classes for reinforcement learning,
including data structures, mathematical operations, and helper functions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import deque, defaultdict
import random
import matplotlib.pyplot as plt
from scipy import stats
import time

class ReplayBuffer:
    """Experience replay buffer for RL agents"""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.size = 0
        self.ptr = 0

    def push(self, state: np.ndarray, action: np.ndarray,
             reward: float, next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences"""
        indices = np.random.choice(self.size, batch_size, replace=False)

        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.BoolTensor(self.dones[indices])
        }

    def __len__(self) -> int:
        return self.size

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, alpha: float = 0.6):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self.max_priority = 1.0

    def push(self, state: np.ndarray, action: np.ndarray,
             reward: float, next_state: np.ndarray, done: bool):
        """Add experience with max priority"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.priorities[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample batch with priorities"""
        if self.size == 0:
            return {}, np.array([]), np.array([])

        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.BoolTensor(self.dones[indices])
        }

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self) -> int:
        return self.size

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""

    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        """Reset noise to mean"""
        self.state = np.ones(self.size) * self.mu

    def sample(self) -> np.ndarray:
        """Sample from OU process"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

class GaussianNoise:
    """Gaussian exploration noise"""

    def __init__(self, size: int, sigma: float = 0.1):
        self.size = size
        self.sigma = sigma

    def sample(self) -> np.ndarray:
        """Sample Gaussian noise"""
        return np.random.normal(0, self.sigma, self.size)

    def reset(self):
        """No-op for Gaussian noise"""
        pass

class RunningNormalizer:
    """Running statistics normalizer"""

    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-8):
        self.shape = shape
        self.eps = eps
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = eps

    def update(self, x: np.ndarray):
        """Update running statistics"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (m_a + m_b + delta**2 * self.count * batch_count / total_count)

        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using running statistics"""
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft update target network parameters"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def hard_update(target: nn.Module, source: nn.Module):
    """Hard update target network parameters"""
    target.load_state_dict(source.state_dict())

def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                next_values: torch.Tensor, dones: torch.Tensor,
                gamma: float = 0.99, lambda_gae: float = 0.95) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE)"""
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_values[t]
        else:
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_gae * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return torch.FloatTensor(advantages)

def compute_returns(rewards: torch.Tensor, dones: torch.Tensor,
                   gamma: float = 0.99) -> torch.Tensor:
    """Compute discounted returns"""
    returns = []
    R = 0

    for r, done in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1 - done)
        returns.insert(0, R)

    return torch.FloatTensor(returns)

def plot_learning_curve(rewards: List[float], window: int = 10,
                       title: str = "Learning Curve", save_path: Optional[str] = None):
    """Plot learning curve with smoothing"""
    if len(rewards) < window:
        smoothed_rewards = rewards
    else:
        smoothed_rewards = []
        for i in range(len(rewards) - window + 1):
            smoothed_rewards.append(np.mean(rewards[i:i + window]))

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw')
    plt.plot(range(window - 1, len(rewards)), smoothed_rewards,
             color='blue', linewidth=2, label=f'Smoothed (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_multiple_curves(curves: Dict[str, List[float]], window: int = 10,
                        title: str = "Comparison", save_path: Optional[str] = None):
    """Plot multiple learning curves for comparison"""
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i, (name, rewards) in enumerate(curves.items()):
        if len(rewards) < window:
            smoothed = rewards
            x_vals = range(len(rewards))
        else:
            smoothed = []
            for j in range(len(rewards) - window + 1):
                smoothed.append(np.mean(rewards[j:j + window]))
            x_vals = range(window - 1, len(rewards))

        color = colors[i % len(colors)]
        plt.plot(rewards, alpha=0.3, color=color)
        plt.plot(x_vals, smoothed, color=color, linewidth=2, label=name)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compute_metrics(rewards: List[float], window: int = 100) -> Dict[str, float]:
    """Compute performance metrics"""
    if len(rewards) == 0:
        return {}

    metrics = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'median_reward': np.median(rewards)
    }

    if len(rewards) >= window:
        metrics['final_avg_reward'] = np.mean(rewards[-window:])
        metrics['final_std_reward'] = np.std(rewards[-window:])

        prev_window = rewards[-(2*window):-window]
        curr_window = rewards[-window:]
        if len(prev_window) == window:
            metrics['convergence_improvement'] = np.mean(curr_window) - np.mean(prev_window)

    return metrics

class Timer:
    """Simple timer for performance measurement"""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def start(self):
        """Start timer"""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop timer and return elapsed time"""
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed

    def reset(self):
        """Reset timer"""
        self.start_time = None
        self.elapsed = 0

    def get_elapsed(self) -> float:
        """Get elapsed time without stopping"""
        if self.start_time is not None:
            return time.time() - self.start_time
        return self.elapsed

class Config:
    """Configuration class for experiments"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return '\n'.join(f'{k}: {v}' for k, v in self.__dict__.items())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config_dict)

def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device(device: Optional[str] = None) -> torch.device:
    """Get torch device"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

print("✅ Utilities module complete!")
print("Components implemented:")
print("- ReplayBuffer: Experience replay")
print("- PrioritizedReplayBuffer: Prioritized experience replay")
print("- OUNoise/GaussianNoise: Exploration noise")
print("- RunningNormalizer: Online normalization")
print("- soft_update/hard_update: Network updates")
print("- compute_gae/compute_returns: Advantage estimation")
print("- plot_learning_curve/plot_multiple_curves: Visualization")
print("- compute_metrics: Performance evaluation")
print("- Timer: Performance timing")
print("- Config: Configuration management")
print("- set_random_seed: Reproducibility")
print("- get_device: Device management")
