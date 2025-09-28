"""
Data Structures for Reinforcement Learning

This module contains data structures used across RL algorithms.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import numpy as np
from collections import namedtuple


# Named tuples for better code readability
Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)
Trajectory = namedtuple("Trajectory", ["states", "actions", "rewards", "dones"])


class Dataset:
    """Dataset for storing and sampling transitions."""

    def __init__(self):
        """Initialize empty dataset."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the dataset.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def add_transition(self, transition):
        """Add a Transition namedtuple to the dataset.

        Args:
            transition: Transition namedtuple
        """
        self.add(*transition)

    def add_trajectory(self, trajectory):
        """Add a complete trajectory to the dataset.

        Args:
            trajectory: Trajectory namedtuple or dict with states, actions, rewards, dones
        """
        if isinstance(trajectory, Trajectory):
            states = trajectory.states
            actions = trajectory.actions
            rewards = trajectory.rewards
            dones = trajectory.dones
        else:
            states = trajectory["states"]
            actions = trajectory["actions"]
            rewards = trajectory["rewards"]
            dones = trajectory["dones"]

        # Convert to transitions
        for i in range(len(states)):
            next_state = states[i + 1] if i + 1 < len(states) else states[i]
            done = dones[i] if i < len(dones) else True
            self.add(states[i], actions[i], rewards[i], next_state, done)

    def size(self):
        """Return dataset size."""
        return len(self.states)

    def is_empty(self):
        """Check if dataset is empty."""
        return self.size() == 0

    def get_all(self):
        """Get all data as numpy arrays."""
        return {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "next_states": np.array(self.next_states),
            "dones": np.array(self.dones),
        }

    def sample(self, batch_size, replace=True):
        """Sample a batch of transitions.

        Args:
            batch_size: Size of batch to sample
            replace: Whether to sample with replacement

        Returns:
            Dictionary of batched data
        """
        if self.is_empty():
            return None

        n_samples = min(batch_size, self.size())
        if replace:
            indices = np.random.randint(0, self.size(), n_samples)
        else:
            indices = np.random.choice(self.size(), n_samples, replace=False)

        return {
            "states": np.array(self.states)[indices],
            "actions": np.array(self.actions)[indices],
            "rewards": np.array(self.rewards)[indices],
            "next_states": np.array(self.next_states)[indices],
            "dones": np.array(self.dones)[indices],
        }

    def clear(self):
        """Clear all data."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

    def get_statistics(self):
        """Get dataset statistics."""
        if self.is_empty():
            return {}

        return {
            "size": self.size(),
            "state_mean": np.mean(self.states, axis=0),
            "state_std": np.std(self.states, axis=0),
            "action_mean": np.mean(self.actions, axis=0),
            "action_std": np.std(self.actions, axis=0),
            "reward_mean": np.mean(self.rewards),
            "reward_std": np.std(self.rewards),
            "episode_count": np.sum(self.dones),
        }


class RollingDataset(Dataset):
    """Dataset with rolling window (fixed maximum size)."""

    def __init__(self, max_size=100000):
        """Initialize rolling dataset.

        Args:
            max_size: Maximum number of transitions to keep
        """
        super().__init__()
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        """Add transition with size management."""
        super().add(state, action, reward, next_state, done)

        # Remove oldest transitions if over limit
        while self.size() > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
