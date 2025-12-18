from __future__ import annotations
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SyntheticBanditDataset(Dataset):
    """
    A tiny synthetic dataset that mimics (state, action, reward, constraint) tuples.
    Used for unit tests and quick debug runs.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        obs_dim: int = 8,
        action_dim: int = 2,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        self.obs = rng.randn(num_samples, obs_dim).astype("float32")
        self.actions = rng.randn(num_samples, action_dim).astype("float32")
        # reward correlated with first observation feature
        self.rewards = (self.obs[:, 0] + 0.1 * rng.randn(num_samples)).astype("float32")
        # constraint: positive when obs[:,1] > 0 (violations)
        self.constraints = (self.obs[:, 1] > 0).astype("float32")

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "action": torch.from_numpy(self.actions[idx]),
            "reward": torch.tensor(self.rewards[idx]),
            "constraint": torch.tensor(self.constraints[idx]),
        }


def make_dataloader(batch_size: int = 64, **kwargs) -> DataLoader:
    ds = SyntheticBanditDataset(**kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)









