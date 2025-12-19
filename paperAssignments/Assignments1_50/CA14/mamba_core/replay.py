"""Simple replay buffer storing contiguous trajectory segments.

Designed for unit tests and small-scale experiments. Not highly optimized.
"""

from typing import Deque, Dict, List, Tuple
from collections import deque
import random

import numpy as np


class ReplayBuffer:
    def __init__(
        self, obs_dim: int, act_dim: int, capacity: int = 100000, seq_len: int = 50
    ):
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.buffer: Deque[Dict] = deque(maxlen=capacity)

    def add_episode(
        self,
        obs: List[np.ndarray],
        acts: List[np.ndarray],
        rews: List[float],
        dones: List[bool],
        morph_id: str = "",
    ):
        """Add an episode as lists/arrays of length L.
        Stores as a dict; sampling will extract contiguous windows.
        """
        self.buffer.append(
            {
                "obs": np.asarray(obs),
                "acts": np.asarray(acts),
                "rews": np.asarray(rews),
                "dones": np.asarray(dones),
                "morph_id": morph_id,
            }
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch_size segments of length seq_len. Pads short episodes by repeating last frame.
        Returns dict with keys: obs (B, L, D), acts (B, L, A), rews (B, L), dones (B, L)
        """
        obs_batch = []
        act_batch = []
        rew_batch = []
        done_batch = []
        for _ in range(batch_size):
            ep = random.choice(list(self.buffer))
            L = len(ep["obs"])
            if L >= self.seq_len:
                start = random.randint(0, L - self.seq_len)
                obs_seg = ep["obs"][start : start + self.seq_len]
                act_seg = ep["acts"][start : start + self.seq_len]
                rew_seg = ep["rews"][start : start + self.seq_len]
                done_seg = ep["dones"][start : start + self.seq_len]
            else:
                # pad
                pad = self.seq_len - L
                obs_seg = np.concatenate(
                    [ep["obs"], np.repeat(ep["obs"][-1:], [pad], axis=0)]
                )
                act_seg = np.concatenate(
                    [ep["acts"], np.repeat(ep["acts"][-1:], [pad], axis=0)]
                )
                rew_seg = np.concatenate([ep["rews"], np.zeros(pad)])
                done_seg = np.concatenate([ep["dones"], np.ones(pad)])
            obs_batch.append(obs_seg)
            act_batch.append(act_seg)
            rew_batch.append(rew_seg)
            done_batch.append(done_seg)
        return {
            "obs": np.asarray(obs_batch),
            "acts": np.asarray(act_batch),
            "rews": np.asarray(rew_batch),
            "dones": np.asarray(done_batch),
        }















