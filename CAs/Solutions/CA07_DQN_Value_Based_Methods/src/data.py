import numpy as np
from collections import deque
import random
import torch
from typing import Tuple, List

class ReplayBuffer:
    """
    Experience replay buffer for DQN agents.

    Stores and samples experiences (state, action, reward, next_state, done)
    to break correlations in the training data and improve stability.
    """

    def __init__(self, capacity: int):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity: The maximum number of experiences to store in the buffer.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Adds an experience tuple to the buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: A boolean indicating if the episode terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Samples a random batch of experiences from the buffer.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A tuple of Torch tensors: (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        """
        Returns the current number of experiences in the buffer.
        """
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer for DQN agents.

    Samples experiences based on their TD-error, giving higher priority
    to more 'surprising' transitions. Uses sum-tree data structure for efficient sampling.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initializes the PrioritizedReplayBuffer.

        Args:
            capacity: The maximum number of experiences to store in the buffer.
            alpha: Prioritization exponent. 0 for uniform, 1 for greedy prioritization.
        """
        self.alpha = alpha
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0  # Initial max priority

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Adds an experience tuple to the buffer with maximum priority.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: A boolean indicating if the episode terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)  # New samples get max priority

    def sample(self, batch_size: int, beta: float = 0.4, epsilon: float = 1e-6) -> Tuple[torch.Tensor, ...]:
        """
        Samples a batch of experiences using prioritized sampling.

        Args:
            batch_size: The number of experiences to sample.
            beta: Importance sampling exponent (0 for no correction, 1 for full correction).
            epsilon: Small constant to add to TD-error to avoid zero priority.

        Returns:
            A tuple of: (states, actions, rewards, next_states, dones, weights, indices).
        """
        if len(self.buffer) < batch_size:
            raise ValueError("Buffer has fewer elements than batch_size for sampling.")

        # Calculate sampling probabilities
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sum_scaled_priorities = scaled_priorities.sum()
        sampling_probs = scaled_priorities / sum_scaled_priorities

        # Sample indices
        indices = random.choices(range(len(self.buffer)), weights=sampling_probs, k=batch_size)

        # Retrieve experiences and calculate importance sampling weights
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])

        # Calculate importance sampling weights
        max_is_weight = (len(self.buffer) * np.min(sampling_probs[indices])) ** (-beta)
        weights = (
            (len(self.buffer) * sampling_probs[indices]) ** (-beta) / max_is_weight
        )

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            torch.FloatTensor(weights),
            indices,
        )

    def update_priorities(self, indices: List[int], td_errors: np.ndarray, epsilon: float = 1e-6):
        """
        Updates priorities of sampled experiences.

        Args:
            indices: List of indices of the sampled experiences.
            td_errors: Corresponding TD-errors for the sampled experiences.
            epsilon: Small constant to add to TD-error to avoid zero priority.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + epsilon
            self.priorities[idx] = priority
        self.max_priority = max(self.priorities)

    def __len__(self) -> int:
        """
        Returns the current number of experiences in the buffer.
        """
        return len(self.buffer)
