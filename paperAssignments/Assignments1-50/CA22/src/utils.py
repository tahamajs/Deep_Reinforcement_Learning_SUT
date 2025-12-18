from __future__ import annotations
from typing import Any, Dict
import random
import os
import json
import torch


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch seeds for reproducibility."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Any | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """Save model and optimizer state to a file (atomic write)."""
    temp_path = path + ".tmp"
    data = {"model_state": model.state_dict()}
    if optimizer is not None:
        try:
            data["optim_state"] = optimizer.state_dict()
        except Exception:
            data["optim_state"] = None
    if extra is not None:
        data["extra"] = extra
    torch.save(data, temp_path)
    os.replace(temp_path, path)


def load_checkpoint(
    path: str, model: torch.nn.Module, optimizer: Any | None = None
) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data.get("model_state", {}))
    if optimizer is not None and data.get("optim_state") is not None:
        try:
            optimizer.load_state_dict(data["optim_state"])
        except Exception:
            # incompatible optimizer state
            pass
    return data


def update_lagrange(
    mu: float, constraint_value: float, c: float, lr: float, max_mu: float
) -> float:
    """Simple projected gradient ascent update for Lagrange multiplier.

    mu <- max(0, mu + lr * (constraint_value - c)) then clip to max_mu
    """
    new_mu = float(mu) + float(lr) * (float(constraint_value) - float(c))
    new_mu = max(0.0, new_mu)
    new_mu = min(new_mu, float(max_mu))
    return new_mu









