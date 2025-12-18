from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """
    Small synthetic dataset useful for unit tests and quick debug runs.

    Each sample is a tuple (observation, action, reward, next_observation, done).
    Observations are float vectors; actions are integer indices.
    """

    def __init__(
        self,
        num_samples: int = 256,
        input_dim: int = 8,
        action_dim: int = 4,
        seed: Optional[int] = None,
    ):
        super().__init__()
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.observations = torch.randn(num_samples, input_dim, generator=rng)
        self.actions = torch.randint(
            low=0, high=action_dim, size=(num_samples,), generator=rng
        )
        self.rewards = torch.randn(
            num_samples,
        )
        self.next_observations = torch.randn(num_samples, input_dim, generator=rng)
        self.dones = torch.zeros(num_samples, dtype=torch.bool)

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_observations[idx],
            self.dones[idx],
        )













