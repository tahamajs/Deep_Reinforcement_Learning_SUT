"""
Replay Buffer Implementation

This module contains various replay buffer implementations for different RL algorithms.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, max_size=1000000):
        """Initialize replay buffer.

        Args:
            max_size: Maximum buffer size
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, *args):
        """Add experience to buffer.

        Args:
            *args: Experience tuple (state, action, reward, next_state, done, ...)
        """
        self.buffer.append(args)

    def sample(self, batch_size):
        """Sample batch of experiences.

        Args:
            batch_size: Size of batch to sample

        Returns:
            Dictionary of batched experiences
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        # Unzip the batch
        batch_arrays = list(zip(*batch))

        return {
            "states": np.array(batch_arrays[0]),
            "actions": np.array(batch_arrays[1]),
            "rewards": np.array(batch_arrays[2]),
            "next_states": np.array(batch_arrays[3]),
            "dones": np.array(batch_arrays[4]),
        }

    def size(self):
        """Return current buffer size."""
        return len(self.buffer)

    def is_full(self):
        """Check if buffer is full."""
        return len(self.buffer) == self.max_size

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer."""

    def __init__(self, max_size=1000000, alpha=0.6, beta=0.4, epsilon=1e-6):
        """Initialize prioritized replay buffer.

        Args:
            max_size: Maximum buffer size
            alpha: Prioritization exponent
            beta: Importance sampling exponent
            epsilon: Small constant for priorities
        """
        super().__init__(max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add(self, *args):
        """Add experience with maximum priority."""
        super().add(*args)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        """Sample batch with prioritization.

        Args:
            batch_size: Size of batch to sample

        Returns:
            Dictionary with experiences and importance weights
        """
        if len(self.buffer) == 0:
            return None

        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]

        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Unzip the batch
        batch_arrays = list(zip(*batch))

        return {
            "states": np.array(batch_arrays[0]),
            "actions": np.array(batch_arrays[1]),
            "rewards": np.array(batch_arrays[2]),
            "next_states": np.array(batch_arrays[3]),
            "dones": np.array(batch_arrays[4]),
            "weights": weights,
            "indices": indices,
        }

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            self.max_priority = max(self.max_priority, priority + self.epsilon)


class TrajectoryBuffer:
    """Buffer for storing complete trajectories."""

    def __init__(self, max_size=1000):
        """Initialize trajectory buffer.

        Args:
            max_size: Maximum number of trajectories to store
        """
        self.buffer = deque(maxlen=max_size)
        self.current_trajectory = []

    def add_step(self, state, action, reward, next_state, done):
        """Add a step to the current trajectory.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.current_trajectory.append((state, action, reward, next_state, done))

        if done:
            self.buffer.append(self.current_trajectory)
            self.current_trajectory = []

    def get_trajectory(self, index=-1):
        """Get a trajectory by index.

        Args:
            index: Trajectory index (-1 for most recent)

        Returns:
            Trajectory as list of (state, action, reward, next_state, done) tuples
        """
        if len(self.buffer) == 0:
            return []
        return self.buffer[index]

    def sample_trajectory(self):
        """Sample a random trajectory."""
        if len(self.buffer) == 0:
            return []
        return random.choice(self.buffer)

    def size(self):
        """Return number of stored trajectories."""
        return len(self.buffer)

    def clear(self):
        """Clear all trajectories."""
        self.buffer.clear()
        self.current_trajectory = []
