from typing import Deque, Tuple, Optional
from collections import deque
import numpy as np
import torch


class SequenceReplayBuffer:
    """
    Minimal replay buffer that stores fixed-length sequences for testing and
    development. Intended for small-scale smoke tests; not optimized.
    """

    def __init__(
        self, obs_dim: int, action_dim: int, seq_len: int, max_size: int = 1000
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.max_size = max_size
        self._buf: Deque = deque(maxlen=max_size)

    def add(
        self,
        obs_seq: np.ndarray,
        act_seq: np.ndarray,
        rew_seq: np.ndarray,
        done_seq: np.ndarray,
        beh_logp: Optional[np.ndarray] = None,
    ):
        """
        Add a single sequence to the buffer. Arrays expected shapes:
            obs_seq: [L, obs_dim]
            act_seq: [L, action_dim]
            rew_seq: [L]
            done_seq: [L]
            beh_logp: [L] or None
        """
        self._buf.append(
            (
                obs_seq.astype(np.float32),
                act_seq.astype(np.float32),
                rew_seq.astype(np.float32),
                done_seq.astype(np.float32),
                None if beh_logp is None else beh_logp.astype(np.float32),
            )
        )

    def __len__(self):
        return len(self._buf)

    def sample_batch(self, batch_size: int, device: str = "cpu"):
        """
        Sample a random batch of sequences and return tensors suitable for training.
        Returns:
            obs: [B, L, obs_dim], acts: [B, L, action_dim], rewards: [B, L], dones: [B, L], beh_logp: [B, L] or None
        """
        idx = np.random.choice(
            len(self._buf), size=batch_size, replace=len(self._buf) < batch_size
        )
        obs, acts, rews, dones, logps = zip(*[self._buf[i] for i in idx])
        obs = torch.tensor(np.stack(obs, axis=0), device=device)
        acts = torch.tensor(np.stack(acts, axis=0), device=device)
        rews = torch.tensor(np.stack(rews, axis=0), device=device)
        dones = torch.tensor(np.stack(dones, axis=0), device=device)
        if logps[0] is None:
            return obs, acts, rews, dones, None
        else:
            return (
                obs,
                acts,
                rews,
                dones,
                torch.tensor(np.stack(logps, axis=0), device=device),
            )


