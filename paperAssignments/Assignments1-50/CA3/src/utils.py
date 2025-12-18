from typing import Any, Dict, Iterable, List, Tuple
import os
import json
import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across torch, numpy and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Save a checkpoint dict to `path`. Creates parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save PyTorch tensors via torch.save for full fidelity
    torch.save(state, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a checkpoint saved by save_checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def discounted_returns(rewards: Iterable[float], gamma: float) -> List[float]:
    """Compute discounted returns G_t for a sequence of rewards.

    Returns a list of the same length as rewards where G_t = r_t + gamma * r_{t+1} + ...
    """
    R = 0.0
    returns = []
    for r in reversed(list(rewards)):
        R = r + gamma * R
        returns.append(R)
    returns = list(reversed(returns))
    return returns


def returns_to_tensor(returns: List[float], device: str = "cpu") -> torch.Tensor:
    """Convert returns list to a float tensor on device."""
    return torch.as_tensor(returns, dtype=torch.float32, device=device)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


