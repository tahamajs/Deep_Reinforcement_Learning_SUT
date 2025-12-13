import random
import torch
import numpy as np
import math
import time

from collections import deque
from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set torch to deterministic mode if desired
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(device_name: Optional[str] = None) -> torch.device:
    """Set the global device for PyTorch operations."""
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print(f"Using device: {device}")
    return device


def get_device() -> torch.device:
    """Get the current default PyTorch device."""
    return torch.tensor(0.).device


def to_tensor(data: Any, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array or list to torch tensor."""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype).to(get_device())
    elif isinstance(data, list):
        return torch.tensor(data, dtype=dtype).to(get_device())
    elif isinstance(data, torch.Tensor):
        return data.to(dtype).to(get_device())
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


class ReplayBuffer:
    """Experience replay buffer for storing transitions."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a new transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Any] = []
        self.priorities: List[float] = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, priority: Optional[float] = None):
        """Add a new transition to the buffer with an optional priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None) # type: ignore
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = (
            priority if priority is not None else max(self.priorities) if self.priorities else 1.0
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample a batch of transitions with importance sampling weights."""
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty.")

        priorities = np.array(self.priorities[: len(self.buffer)])
        probabilities = priorities**self.alpha
        probabilities = probabilities / probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        return samples, indices, to_tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class RunningStats:
    """Running statistics for normalization."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, x):
        """Update statistics with a new data point."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
        self.min = min(self.min, x)
        self.max = max(self.max, x)

    @property
    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self):
        return math.sqrt(self.variance)


@dataclass
class EpisodeMetrics:
    """Container for per-episode statistics across training loops."""

    episode: int
    return_: float
    length: int
    elapsed_sec: float
    mean_q_loss: Optional[float] = None
    mean_model_loss: Optional[float] = None
    mean_planning_reward: Optional[float] = None
    success: Optional[bool] = None
    final_distance: Optional[float] = None
    notes: Dict[str, Any] = field(default_factory=dict)


def env_reset(env: Any):
    """Helper for Gymnasium environment reset compatibility."""
    # Handle Gymnasium API change (v0.26+)
    if hasattr(env, "_max_episode_steps"):
        obs, info = env.reset()
        return obs
    else:
        return env.reset()


def env_step(env: Any, action: Any):
    """Helper for Gymnasium environment step compatibility."""
    # Handle Gymnasium API change (v0.26+)
    if hasattr(env, "_max_episode_steps"):
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    else:
        return env.step(action)


