from typing import Iterator, Tuple
import torch
from torch.utils.data import Dataset


class RandomTrajectoryDataset(Dataset):
    """
    Simple dataset that yields random trajectories for development and tests.
    This is intentionally lightweight and import-safe.
    """

    def __init__(self, seq_len: int, d_model: int, size: int = 1000):
        self.seq_len = seq_len
        self.d_model = d_model
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # observations: (seq_len, d_model), actions: (seq_len, d_model)
        obs = torch.randn(self.seq_len, self.d_model, dtype=torch.float32)
        actions = torch.randn(self.seq_len, self.d_model, dtype=torch.float32)
        return obs, actions














