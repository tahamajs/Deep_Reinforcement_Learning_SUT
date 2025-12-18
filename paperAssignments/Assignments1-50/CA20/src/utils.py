from __future__ import annotations
import os
import random
import json
from typing import Any, Dict
import torch
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: Any, extra: Dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Any | None = None) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload.get("extra", {})


class LagrangeMultiplier:
    """Simple stateful Lagrange multiplier with projected gradient ascent update."""

    def __init__(self, initial: float = 1.0, lr: float = 1e-2, clip: float = 1e6):
        self.value = float(initial)
        self.lr = float(lr)
        self.clip = float(clip)

    def step(self, constraint_estimate: float, threshold: float) -> float:
        """
        Update rule: mu <- max(0, mu + lr * (constraint - threshold))
        """
        update = self.lr * (constraint_estimate - threshold)
        self.value += update
        # projection to non-negative and clipping for stability
        self.value = max(0.0, min(self.value, self.clip))
        return self.value

    def state_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "lr": self.lr, "clip": self.clip}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.value = float(d.get("value", self.value))
        self.lr = float(d.get("lr", self.lr))
        self.clip = float(d.get("clip", self.clip))

