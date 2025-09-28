"""
Normalization Utilities

This module provides normalization utilities for RL training.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import numpy as np


class Normalizer:
    """Running statistics normalizer."""

    def __init__(self, shape, eps=1e-8):
        """Initialize normalizer.

        Args:
            shape: Shape of data to normalize
            eps: Small epsilon for numerical stability
        """
        self.shape = shape
        self.eps = eps

        # Running statistics
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.std = np.ones(shape)
        self.count = 0

    def update(self, data):
        """Update running statistics with new data.

        Args:
            data: New data batch (batch_size, ...)
        """
        if len(data) == 0:
            return

        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        batch_mean = np.mean(data, axis=0)
        batch_var = np.var(data, axis=0)
        batch_count = len(data)

        # Update running statistics
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Update mean
        self.mean = (self.count * self.mean + batch_count * batch_mean) / total_count

        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m_2 / total_count

        self.count = total_count
        self.std = np.sqrt(self.var + self.eps)

    def normalize(self, data):
        """Normalize data using running statistics.

        Args:
            data: Data to normalize

        Returns:
            Normalized data
        """
        return (data - self.mean) / self.std

    def unnormalize(self, data):
        """Unnormalize data.

        Args:
            data: Normalized data

        Returns:
            Unnormalized data
        """
        return data * self.std + self.mean

    def get_stats(self):
        """Get current statistics."""
        return {
            "mean": self.mean.copy(),
            "std": self.std.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def reset(self):
        """Reset statistics."""
        self.mean = np.zeros(self.shape)
        self.var = np.ones(self.shape)
        self.std = np.ones(self.shape)
        self.count = 0


class RewardNormalizer(Normalizer):
    """Reward normalizer with return-based normalization."""

    def __init__(self, gamma=0.99, eps=1e-8):
        """Initialize reward normalizer.

        Args:
            gamma: Discount factor for return calculation
            eps: Small epsilon
        """
        super().__init__((1,), eps)
        self.gamma = gamma
        self.returns = []

    def update(self, rewards):
        """Update with episode rewards.

        Args:
            rewards: Episode rewards (list or array)
        """
        # Calculate discounted return
        returns = []
        discounted_return = 0

        for reward in reversed(rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)

        self.returns.extend(returns)
        super().update(self.returns)

    def normalize_reward(self, reward):
        """Normalize single reward.

        Args:
            reward: Reward value

        Returns:
            Normalized reward
        """
        return self.normalize(np.array([reward]))[0]

    def normalize_rewards(self, rewards):
        """Normalize reward sequence.

        Args:
            rewards: Reward sequence

        Returns:
            Normalized rewards
        """
        return self.normalize(np.array(rewards))


class StateNormalizer(Normalizer):
    """State/observation normalizer."""

    def __init__(self, obs_shape, clip_range=(-5, 5), eps=1e-8):
        """Initialize state normalizer.

        Args:
            obs_shape: Observation shape
            clip_range: Range to clip normalized values
            eps: Small epsilon
        """
        super().__init__(obs_shape, eps)
        self.clip_range = clip_range

    def normalize(self, obs):
        """Normalize observation.

        Args:
            obs: Observation

        Returns:
            Normalized observation
        """
        normalized = super().normalize(obs)
        return np.clip(normalized, self.clip_range[0], self.clip_range[1])


class ActionNormalizer:
    """Action normalizer for continuous action spaces."""

    def __init__(self, action_space):
        """Initialize action normalizer.

        Args:
            action_space: Gym action space
        """
        self.low = action_space.low
        self.high = action_space.high
        self.range = self.high - self.low

    def normalize(self, action):
        """Normalize action to [-1, 1].

        Args:
            action: Raw action

        Returns:
            Normalized action
        """
        return 2 * (action - self.low) / self.range - 1

    def unnormalize(self, action):
        """Unnormalize action from [-1, 1] to original range.

        Args:
            action: Normalized action

        Returns:
            Unnormalized action
        """
        return self.low + (action + 1) * self.range / 2
