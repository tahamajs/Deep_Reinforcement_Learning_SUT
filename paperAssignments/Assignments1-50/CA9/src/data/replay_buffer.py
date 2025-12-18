from typing import Dict, Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Simple numpy-backed replay buffer for offline datasets and online additions."""

    def __init__(self, max_size: int = 1_000_000):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.s = None
        self.a = None
        self.r = None
        self.s_next = None
        self.done = None

    def load_from_arrays(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        rews: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ):
        """Initialize buffer from numpy arrays (N, ...)."""
        N = obs.shape[0]
        self.max_size = max(self.max_size, N)
        self.s = np.zeros((self.max_size,) + obs.shape[1:], dtype=np.float32)
        self.a = np.zeros((self.max_size,) + acts.shape[1:], dtype=np.float32)
        self.r = np.zeros((self.max_size,), dtype=np.float32)
        self.s_next = np.zeros((self.max_size,) + next_obs.shape[1:], dtype=np.float32)
        self.done = np.zeros((self.max_size,), dtype=np.float32)
        self.s[:N] = obs
        self.a[:N] = acts
        self.r[:N] = rews
        self.s_next[:N] = next_obs
        self.done[:N] = dones
        self.size = N
        self.ptr = N % self.max_size

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: float,
    ):
        if self.s is None:
            # lazy init
            self.s = np.zeros((self.max_size,) + obs.shape, dtype=np.float32)
            self.a = np.zeros((self.max_size,) + act.shape, dtype=np.float32)
            self.r = np.zeros((self.max_size,), dtype=np.float32)
            self.s_next = np.zeros((self.max_size,) + next_obs.shape, dtype=np.float32)
            self.done = np.zeros((self.max_size,), dtype=np.float32)
        self.s[self.ptr] = obs
        self.a[self.ptr] = act
        self.r[self.ptr] = rew
        self.s_next[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 256) -> Dict[str, torch.Tensor]:
        assert self.size > 0, "ReplayBuffer is empty"
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            s=torch.from_numpy(self.s[idx]).float(),
            a=torch.from_numpy(self.a[idx]).float(),
            r=torch.from_numpy(self.r[idx]).float(),
            s_next=torch.from_numpy(self.s_next[idx]).float(),
            done=torch.from_numpy(self.done[idx]).float(),
        )
        return batch

    @staticmethod
    def from_d4rl(env_name: str, max_size: Optional[int] = None) -> "ReplayBuffer":
        try:
            import d4rl  # type: ignore
            import gymnasium as gym
        except Exception as e:
            raise RuntimeError(
                "d4rl/gymnasium not available. Install d4rl to load datasets."
            ) from e
        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        obs = dataset["observations"].astype(np.float32)
        acts = dataset["actions"].astype(np.float32)
        rews = dataset["rewards"].astype(np.float32)
        next_obs = dataset["next_observations"].astype(np.float32)
        dones = dataset["terminals"].astype(np.float32)
        rb = ReplayBuffer(max_size=max_size or obs.shape[0])
        rb.load_from_arrays(obs, acts, rews, next_obs, dones)
        return rb












