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









