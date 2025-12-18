from typing import Iterator, Tuple
import numpy as np
import torch


class SyntheticDataset:
    """
    Minimal synthetic dataset producing random states and targets for quick testing.
    Yields tuples (state, action_target, value_target).
    """

    def __init__(
        self, input_dim: int, output_dim: int, size: int = 1024, seed: int = 0
    ) -> None:
        rng = np.random.RandomState(seed)
        self.states = rng.randn(size, input_dim).astype(np.float32)
        # random categorical targets in [0, output_dim)
        self.actions = rng.randint(0, output_dim, size).astype(np.int64)
        # random values
        self.values = rng.randn(size).astype(np.float32)
        self.size = size

    def __len__(self) -> int:
        return self.size

    def batches(
        self, batch_size: int
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        idx = np.arange(self.size)
        rng = np.random.RandomState(0)
        rng.shuffle(idx)
        for i in range(0, self.size, batch_size):
            batch_idx = idx[i : i + batch_size]
            s = torch.from_numpy(self.states[batch_idx])
            a = torch.from_numpy(self.actions[batch_idx])
            v = torch.from_numpy(self.values[batch_idx])
            yield s, a, v













