from __future__ import annotations
import random
from typing import Optional, Tuple, List, Dict
import torch
import os


class CheckpointBuffer:
    """
    Fixed-capacity buffer for storing latent checkpoints and metadata.

    Each entry stores a latent tensor `z` (torch.Tensor), a scalar score,
    and the env step at which it was recorded.
    """

    def __init__(self, capacity: int = 1024, device: Optional[torch.device] = None):
        self.capacity = int(capacity)
        self.device = device or torch.device("cpu")
        self._z_store: List[Optional[torch.Tensor]] = [None] * self.capacity
        self._score: List[float] = [0.0] * self.capacity
        self._step: List[int] = [0] * self.capacity
        self._ptr = 0
        self._size = 0

    def push(self, z: torch.Tensor, score: float, step: int) -> None:
        """Push a latent checkpoint (copies tensor to buffer device)."""
        self._z_store[self._ptr] = z.detach().to(self.device).clone()
        self._score[self._ptr] = float(score)
        self._step[self._ptr] = int(step)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, k: int = 1, prioritized: bool = True) -> List[Dict]:
        """
        Sample k checkpoints.
        If prioritized, samples proportional to score (non-negative).
        Returns list of dicts: {"z": Tensor, "score": float, "step": int, "idx": int}
        """
        if self._size == 0:
            return []
        indices = list(range(self._size))
        if prioritized:
            scores = [max(0.0, s) for s in self._score[: self._size]]
            total = sum(scores)
            if total <= 0:
                # fallback to uniform
                chosen = random.sample(indices, min(k, len(indices)))
            else:
                probs = [s / total for s in scores]
                chosen = list(
                    torch.multinomial(
                        torch.tensor(probs), min(k, len(indices)), replacement=False
                    ).tolist()
                )
        else:
            chosen = random.sample(indices, min(k, len(indices)))
        out = []
        for idx in chosen:
            out.append(
                {
                    "z": self._z_store[idx].to(self.device),
                    "score": self._score[idx],
                    "step": self._step[idx],
                    "idx": idx,
                }
            )
        return out

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._z_store = [None] * self.capacity
        self._score = [0.0] * self.capacity
        self._step = [0] * self.capacity
        self._ptr = 0
        self._size = 0

    def save(self, path: str) -> None:
        """Save checkpoints (cpu tensors) to a file (torch.save)."""
        data = {"score": self._score[: self._size], "step": self._step[: self._size]}
        # move tensors to cpu for safe saving
        data["z"] = [
            z.detach().cpu() if z is not None else None
            for z in self._z_store[: self._size]
        ]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        zs = data.get("z", [])
        self.clear()
        for z, s, st in zip(zs, data.get("score", []), data.get("step", [])):
            if z is None:
                continue
            self.push(z.to(self.device), float(s), int(st))

    def to(self, device: torch.device) -> "CheckpointBuffer":
        """Move stored tensors to device in-place."""
        self.device = device
        for i in range(self._size):
            if self._z_store[i] is not None:
                self._z_store[i] = self._z_store[i].to(device)
        return self













