from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticRegressionDataset(Dataset):
    """A tiny synthetic regression dataset used for smoke tests and demo runs.

    This is deterministic (seeded) and import-safe.
    """

    def __init__(self, num_samples: int = 512, input_dim: int = 10, seed: int = 42):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.x = torch.randn((num_samples, input_dim), generator=rng)
        # simple linear targets with noise
        true_w = torch.linspace(0.1, 1.0, input_dim)
        self.y = self.x @ true_w + 0.1 * torch.randn(num_samples, generator=rng)
        self.y = self.y.unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def get_dataloader(batch_size: int = 32, **dataset_kwargs) -> DataLoader:
    ds = SyntheticRegressionDataset(**dataset_kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
