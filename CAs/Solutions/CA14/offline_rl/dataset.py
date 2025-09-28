"""
Offline Dataset Management for Offline Reinforcement Learning

This module provides utilities for managing offline datasets with different quality levels.
"""

import numpy as np
import torch
from collections import defaultdict
from typing import List, Tuple, Dict, Any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OfflineDataset:
    """Dataset class for offline reinforcement learning."""

    def __init__(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
        dataset_type: str = "mixed",
    ):
        """
        Initialize offline dataset.

        Args:
            states: List of state arrays
            actions: List of actions taken
            rewards: List of rewards received
            next_states: List of next state arrays
            dones: List of done flags
            dataset_type: Type of dataset ('expert', 'mixed', 'random')
        """
        self.states = np.array(states, dtype=np.float32)
        self.actions = np.array(actions, dtype=np.int32)
        self.rewards = np.array(rewards, dtype=np.float32)
        self.next_states = np.array(next_states, dtype=np.float32)
        self.dones = np.array(dones, dtype=bool)
        self.dataset_type = dataset_type

        # Compute statistics
        self.size = len(self.states)
        self.reward_mean = np.mean(self.rewards)
        self.reward_std = np.std(self.rewards)
        self.state_dim = self.states.shape[1] if len(self.states.shape) > 1 else 1
        self.action_dim = int(np.max(self.actions)) + 1

        # Convert to tensors for efficient batching
        self._tensor_data = None

    def get_tensor_data(self) -> Tuple[torch.Tensor, ...]:
        """Get dataset as tensors on the appropriate device."""
        if self._tensor_data is None:
            self._tensor_data = (
                torch.FloatTensor(self.states).to(device),
                torch.LongTensor(self.actions).to(device),
                torch.FloatTensor(self.rewards).to(device),
                torch.FloatTensor(self.next_states).to(device),
                torch.BoolTensor(self.dones).to(device),
            )
        return self._tensor_data

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch from the dataset."""
        indices = np.random.choice(self.size, batch_size, replace=True)
        states, actions, rewards, next_states, dones = self.get_tensor_data()

        return (
            states[indices],
            actions[indices],
            rewards[indices],
            next_states[indices],
            dones[indices],
        )

    def get_action_distribution(self) -> np.ndarray:
        """Get the distribution of actions in the dataset."""
        if self.actions.ndim == 1:  # Discrete actions
            action_counts = np.bincount(self.actions)
            return action_counts / self.size
        else:  # Continuous actions
            return np.mean(self.actions, axis=0), np.std(self.actions, axis=0)

    def filter_by_quality(self, quality_threshold: float) -> "OfflineDataset":
        """Filter dataset by reward quality."""
        high_quality_indices = self.rewards >= quality_threshold
        return OfflineDataset(
            self.states[high_quality_indices],
            self.actions[high_quality_indices],
            self.rewards[high_quality_indices],
            self.next_states[high_quality_indices],
            self.dones[high_quality_indices],
            f"{self.dataset_type}_filtered_{quality_threshold}",
        )

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"OfflineDataset(type={self.dataset_type}, size={self.size}, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"reward_mean={self.reward_mean:.3f})"
        )
