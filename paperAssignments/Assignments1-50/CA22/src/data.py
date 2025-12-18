from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np


def synthetic_episode(
    obs_dim: int = 8,
    horizon: int = 20,
    action_dim: int = 4,
    seed: int | None = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield a single synthetic episode as (states, actions, rewards, constraints).

    - states: (T, obs_dim)
    - actions: (T,)
    - rewards: (T,)
    - constraints: (T,) cost signal (higher is worse)
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.5, size=(obs_dim,))
    Cw = rng.normal(scale=0.3, size=(obs_dim,))

    states = rng.normal(size=(horizon, obs_dim)).astype(np.float32)
    logits = states @ W
    # discrete actions sampled according to softmax over action_dim of projected logits
    # create per-timestep action choices
    actions = rng.integers(0, action_dim, size=(horizon,), dtype=np.int64)

    rewards = (states * W).sum(axis=1) + rng.normal(scale=0.1, size=(horizon,))
    constraints = np.abs((states * Cw).sum(axis=1)) + rng.normal(scale=0.05, size=(horizon,))

    yield states, actions, rewards.astype(np.float32), constraints.astype(np.float32)


class SyntheticDataset:
    """Simple in-memory dataset producing short episodes for debug runs."""

    def __init__(self, num_episodes: int = 100, obs_dim: int = 8, horizon: int = 20, seed: int | None = None):
        self._episodes = []
        rng = np.random.default_rng(seed)
        for i in range(num_episodes):
            ep = next(synthetic_episode(obs_dim=obs_dim, horizon=horizon, seed=rng.integers(0, 2 ** 31 - 1)))
            self._episodes.append(ep)

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int):
        return self._episodes[idx]

    def sample_batch(self, batch_size: int = 32):
        idx = np.random.choice(len(self._episodes), size=batch_size, replace=True)
        batch = [self._episodes[i] for i in idx]
        # flatten across timesteps
        states = np.concatenate([b[0] for b in batch], axis=0)
        actions = np.concatenate([b[1] for b in batch], axis=0)
        rewards = np.concatenate([b[2] for b in batch], axis=0)
        constraints = np.concatenate([b[3] for b in batch], axis=0)
        return states, actions, rewards, constraints
