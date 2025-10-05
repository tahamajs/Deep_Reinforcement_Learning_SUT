"""
Experience Replay for Continual Learning

This module implements experience replay mechanisms for continual learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import random
from collections import deque
import heapq


class Experience:
    """Represents a single experience."""

    def __init__(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        task_id: int,
        priority: float = 1.0,
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.task_id = task_id
        self.priority = priority
        self.timestamp = len(Experience._global_timestamp)
        Experience._global_timestamp.append(self.timestamp)

    def __lt__(self, other):
        return self.priority < other.priority

    def __le__(self, other):
        return self.priority <= other.priority

    def __gt__(self, other):
        return self.priority > other.priority

    def __ge__(self, other):
        return self.priority >= other.priority


# Global timestamp counter
Experience._global_timestamp = []


class ExperienceReplay:
    """Basic experience replay buffer."""

    def __init__(
        self,
        capacity: int = 10000,
        state_dim: int = 4,
        action_dim: int = 2,
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Storage
        self.states = torch.zeros(capacity, state_dim)
        self.actions = torch.zeros(capacity, action_dim)
        self.rewards = torch.zeros(capacity)
        self.next_states = torch.zeros(capacity, state_dim)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        self.task_ids = torch.zeros(capacity, dtype=torch.long)

        # Buffer management
        self.position = 0
        self.size = 0

        # Statistics
        self.total_experiences = 0
        self.task_counts = {}

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        task_id: int,
    ):
        """Add experience to buffer."""
        # Store experience
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.task_ids[self.position] = task_id

        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_experiences += 1

        # Update task counts
        self.task_counts[task_id] = self.task_counts.get(task_id, 0) + 1

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences."""
        if self.size == 0:
            return {}

        indices = torch.randint(0, self.size, (batch_size,))

        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "task_ids": self.task_ids[indices],
        }

    def sample_by_task(self, task_id: int, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample experiences from a specific task."""
        task_indices = torch.where(self.task_ids[: self.size] == task_id)[0]

        if len(task_indices) == 0:
            return {}

        if len(task_indices) < batch_size:
            indices = task_indices
        else:
            indices = task_indices[torch.randint(0, len(task_indices), (batch_size,))]

        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "task_ids": self.task_ids[indices],
        }

    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored experiences."""
        return {
            "total_experiences": self.total_experiences,
            "buffer_size": self.size,
            "buffer_capacity": self.capacity,
            "task_counts": self.task_counts,
            "task_distribution": {
                task_id: count / self.size if self.size > 0 else 0
                for task_id, count in self.task_counts.items()
            },
        }

    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0
        self.total_experiences = 0
        self.task_counts.clear()


class PrioritizedExperienceReplay:
    """Prioritized experience replay buffer."""

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Storage
        self.experiences = []
        self.priorities = []
        self.position = 0

        # Statistics
        self.total_experiences = 0
        self.task_counts = {}

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        task_id: int,
        priority: Optional[float] = None,
    ):
        """Add experience to buffer."""
        if priority is None:
            priority = 1.0

        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "task_id": task_id,
        }

        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
            self.priorities.append(priority)
        else:
            self.experiences[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity
        self.total_experiences += 1
        self.task_counts[task_id] = self.task_counts.get(task_id, 0) + 1

    def sample(
        self, batch_size: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Sample a batch of experiences with priorities."""
        if len(self.experiences) == 0:
            return {}, torch.tensor([]), torch.tensor([])

        # Compute sampling probabilities
        priorities = np.array(self.priorities[: len(self.experiences)])
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.experiences), size=batch_size, p=probabilities
        )

        # Compute importance weights
        weights = (len(self.experiences) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        # Collect experiences
        batch = {
            "states": torch.stack([self.experiences[i]["state"] for i in indices]),
            "actions": torch.stack([self.experiences[i]["action"] for i in indices]),
            "rewards": torch.tensor([self.experiences[i]["reward"] for i in indices]),
            "next_states": torch.stack(
                [self.experiences[i]["next_state"] for i in indices]
            ),
            "dones": torch.tensor([self.experiences[i]["done"] for i in indices]),
            "task_ids": torch.tensor([self.experiences[i]["task_id"] for i in indices]),
        }

        return batch, torch.tensor(weights), torch.tensor(indices)

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority.item()

    def update_beta(self):
        """Update beta for importance sampling."""
        self.beta = min(1.0, self.beta + self.beta_increment)

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "total_experiences": self.total_experiences,
            "buffer_size": len(self.experiences),
            "buffer_capacity": self.capacity,
            "task_counts": self.task_counts,
            "alpha": self.alpha,
            "beta": self.beta,
            "avg_priority": np.mean(self.priorities) if self.priorities else 0.0,
        }


class TaskBalancedReplay:
    """Task-balanced experience replay buffer."""

    def __init__(
        self,
        capacity: int = 10000,
        num_tasks: int = 5,
        balance_factor: float = 0.5,
    ):
        self.capacity = capacity
        self.num_tasks = num_tasks
        self.balance_factor = balance_factor

        # Per-task buffers
        self.task_buffers = [
            deque(maxlen=capacity // num_tasks) for _ in range(num_tasks)
        ]

        # Statistics
        self.total_experiences = 0
        self.task_counts = [0] * num_tasks

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        task_id: int,
    ):
        """Add experience to appropriate task buffer."""
        if 0 <= task_id < self.num_tasks:
            experience = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "task_id": task_id,
            }

        self.task_buffers[task_id].append(experience)
        self.task_counts[task_id] += 1
        self.total_experiences += 1

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample experiences balanced across tasks."""
        if self.total_experiences == 0:
            return {}

        # Compute sampling weights for each task
        task_weights = []
        for i in range(self.num_tasks):
            if self.task_counts[i] > 0:
                # Inverse frequency weighting
                weight = 1.0 / (self.task_counts[i] + 1)
                task_weights.append(weight)
            else:
                task_weights.append(0.0)

        # Normalize weights
        total_weight = sum(task_weights)
        if total_weight > 0:
            task_weights = [w / total_weight for w in task_weights]

        # Sample from each task
        experiences = []
        samples_per_task = batch_size // self.num_tasks
        remaining_samples = batch_size % self.num_tasks

        for i in range(self.num_tasks):
            if self.task_counts[i] > 0:
                # Number of samples for this task
                num_samples = samples_per_task
                if i < remaining_samples:
                    num_samples += 1

                # Sample from task buffer
                task_experiences = list(self.task_buffers[i])
                if len(task_experiences) >= num_samples:
                    sampled = random.sample(task_experiences, num_samples)
                else:
                    sampled = task_experiences

                experiences.extend(sampled)

        if not experiences:
            return {}

        # Convert to batch format
        batch = {
            "states": torch.stack([exp["state"] for exp in experiences]),
            "actions": torch.stack([exp["action"] for exp in experiences]),
            "rewards": torch.tensor([exp["reward"] for exp in experiences]),
            "next_states": torch.stack([exp["next_state"] for exp in experiences]),
            "dones": torch.tensor([exp["done"] for exp in experiences]),
            "task_ids": torch.tensor([exp["task_id"] for exp in experiences]),
        }

        return batch

    def get_task_statistics(self) -> Dict[str, Any]:
        """Get task-balanced statistics."""
        return {
            "total_experiences": self.total_experiences,
            "task_counts": self.task_counts,
            "task_distribution": [
                count / self.total_experiences if self.total_experiences > 0 else 0
                for count in self.task_counts
            ],
            "balance_factor": self.balance_factor,
        }


class ContinualReplay:
    """Continual learning experience replay with forgetting prevention."""

    def __init__(
        self,
        capacity: int = 10000,
        num_tasks: int = 5,
        importance_threshold: float = 0.1,
        forgetting_threshold: float = 0.05,
    ):
        self.capacity = capacity
        self.num_tasks = num_tasks
        self.importance_threshold = importance_threshold
        self.forgetting_threshold = forgetting_threshold

        # Storage
        self.experiences = []
        self.importance_scores = []
        self.task_counts = [0] * num_tasks

        # Forgetting prevention
        self.task_importance = {}
        self.forgetting_rates = {}

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        task_id: int,
        importance: Optional[float] = None,
    ):
        """Add experience with importance scoring."""
        if importance is None:
            importance = 1.0

        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "task_id": task_id,
        }

        # Add to buffer
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
            self.importance_scores.append(importance)
        else:
            # Replace least important experience
            min_importance_idx = np.argmin(self.importance_scores)
            self.experiences[min_importance_idx] = experience
            self.importance_scores[min_importance_idx] = importance

        # Update task counts
        if 0 <= task_id < self.num_tasks:
            self.task_counts[task_id] += 1

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample experiences with importance weighting."""
        if len(self.experiences) == 0:
            return {}

        # Compute sampling probabilities based on importance
        importance_scores = np.array(self.importance_scores)
        probabilities = importance_scores / importance_scores.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.experiences), size=batch_size, p=probabilities
        )

        # Collect experiences
        batch = {
            "states": torch.stack([self.experiences[i]["state"] for i in indices]),
            "actions": torch.stack([self.experiences[i]["action"] for i in indices]),
            "rewards": torch.tensor([self.experiences[i]["reward"] for i in indices]),
            "next_states": torch.stack(
                [self.experiences[i]["next_state"] for i in indices]
            ),
            "dones": torch.tensor([self.experiences[i]["done"] for i in indices]),
            "task_ids": torch.tensor([self.experiences[i]["task_id"] for i in indices]),
        }

        return batch

    def update_importance(self, indices: torch.Tensor, new_importance: torch.Tensor):
        """Update importance scores for experiences."""
        for idx, importance in zip(indices, new_importance):
            if idx < len(self.importance_scores):
                self.importance_scores[idx] = importance.item()

    def detect_forgetting(self, task_id: int, performance: float) -> bool:
        """Detect if forgetting is occurring for a task."""
        if task_id not in self.task_importance:
            self.task_importance[task_id] = []

        self.task_importance[task_id].append(performance)

        # Check for forgetting
        if len(self.task_importance[task_id]) > 10:
            recent_performance = np.mean(self.task_importance[task_id][-5:])
            historical_performance = np.mean(self.task_importance[task_id][-10:-5])

            if recent_performance < historical_performance - self.forgetting_threshold:
                return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get continual replay statistics."""
        return {
            "total_experiences": len(self.experiences),
            "buffer_capacity": self.capacity,
            "task_counts": self.task_counts,
            "avg_importance": (
                np.mean(self.importance_scores) if self.importance_scores else 0.0
            ),
            "task_importance": self.task_importance,
            "forgetting_rates": self.forgetting_rates,
        }
