from typing import Deque, Dict, List, Tuple
from collections import deque

import random
import torch


class ReplayBuffer:
    """A simple in-memory replay buffer for offline sampling.

    Stores transitions as dicts with keys: obs, action, reward, next_obs, done
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self.buffer: Deque[Dict] = deque(maxlen=self.capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
            }
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int):
        """Sample a minibatch of transitions from the buffer.

        Args:
            batch_size: number of transitions to sample (must be <= len(buffer))

        Returns:
            Tuple of tensors: (obs, actions, rewards, next_obs, dones)
                - obs: (B, *obs_shape) float
                - actions: (B,) integer tensor
                - rewards: (B,) float
                - next_obs: (B, *obs_shape) float
                - dones: (B,) float

        Raises:
            ValueError: if batch_size is larger than the current buffer size.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"batch_size ({batch_size}) greater than buffer size ({len(self.buffer)})"
            )
        batch = random.sample(self.buffer, batch_size)
        obs = torch.stack([torch.as_tensor(b["obs"]) for b in batch], dim=0).float()
        actions = torch.as_tensor([b["action"] for b in batch])
        rewards = torch.as_tensor([b["reward"] for b in batch]).float()
        next_obs = torch.stack(
            [torch.as_tensor(b["next_obs"]) for b in batch], dim=0
        ).float()
        dones = torch.as_tensor([b["done"] for b in batch]).float()
        return obs, actions, rewards, next_obs, dones

    def clear(self):
        self.buffer.clear()















