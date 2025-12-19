from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticRegressionDataset(Dataset):
    """A small synthetic regression dataset.

    y = f(x) + noise, where f(x) = sin(2 * pi * x) for x in [0, 1].
    """

    def __init__(self, n_samples: int = 1000, noise_std: float = 0.1, seed: int | None = None):
        self.n_samples = int(n_samples)
        self.noise_std = float(noise_std)
        rng = np.random.RandomState(seed)
        self.x = rng.rand(self.n_samples, 1).astype(np.float32)
        self.y = np.sin(2 * np.pi * self.x) + rng.randn(self.n_samples, 1).astype(np.float32) * self.noise_std

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx])
        y = torch.from_numpy(self.y[idx])
        return x, y


def get_dataloader(n_samples: int = 1000, batch_size: int = 64, seed: int | None = None) -> DataLoader:
    ds = SyntheticRegressionDataset(n_samples=n_samples, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
