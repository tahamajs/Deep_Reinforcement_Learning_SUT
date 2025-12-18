from __future__ import annotations
from typing import Iterator, Tuple
import torch
from torch.utils.data import Dataset


class RandomMDPDataset(Dataset):
    """A tiny synthetic dataset for quick debugging.

    Each item is a dict with keys: obs, action, reward, done, next_obs
    Shapes:
      obs: (obs_dim,)
      action: int
      reward: float
    """

    def __init__(
        self, num_transitions: int = 1024, obs_dim: int = 4, action_dim: int = 2
    ) -> None:
        super().__init__()
        self.num_transitions = num_transitions
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # generate once, small memory
        self.obs = torch.randn(num_transitions, obs_dim)
        self.next_obs = torch.randn(num_transitions, obs_dim)
        self.actions = torch.randint(0, action_dim, (num_transitions,))
        self.rewards = torch.randn(num_transitions)
        self.dones = torch.zeros(num_transitions, dtype=torch.float32)

    def __len__(self) -> int:
        return self.num_transitions

    def __getitem__(self, idx: int):
        return {
            "obs": self.obs[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "done": self.dones[idx],
            "next_obs": self.next_obs[idx],
        }











