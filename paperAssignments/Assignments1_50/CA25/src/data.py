from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticRegressionDataset(Dataset):
    """Synthetic linear-ish regression data with optional noise."""

    def __init__(self, n_samples: int = 1000, input_dim: int = 16, noise: float = 0.1, seed: int = 0):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.X = torch.randn(n_samples, input_dim, generator=rng)
        w = torch.randn(input_dim, 1, generator=rng)
        self.y = (self.X @ w).squeeze(-1) + noise * torch.randn(n_samples, generator=rng)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SyntheticClassificationDataset(Dataset):
    """Synthetic classification data (binary by default)."""

    def __init__(self, n_samples: int = 1000, input_dim: int = 16, n_classes: int = 2, seed: int = 0):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.X = torch.randn(n_samples, input_dim, generator=rng)
        self.logits = torch.randn(input_dim, n_classes, generator=rng)
        self.y = (self.X @ self.logits).argmax(dim=-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], int(self.y[idx])


def get_dataloader(task: str = "regression", batch_size: int = 64, input_dim: int = 16, seed: int = 0) -> Tuple[DataLoader, DataLoader]:
    if task == "classification":
        ds = SyntheticClassificationDataset(n_samples=1000, input_dim=input_dim, seed=seed)
    else:
        ds = SyntheticRegressionDataset(n_samples=1000, input_dim=input_dim, seed=seed)
    n = len(ds)
    split = int(0.8 * n)
    train_ds = torch.utils.data.Subset(ds, list(range(0, split)))
    val_ds = torch.utils.data.Subset(ds, list(range(split, n)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
