from __future__ import annotations
import random
import numpy as np
import torch
from pathlib import Path
from typing import Any, Tuple


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_hint: str = "cpu") -> torch.device:
    if device_hint == "cpu":
        return torch.device("cpu")
    if device_hint == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(state: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def assert_shape(tensor: torch.Tensor, expected: Tuple[int, ...]) -> None:
    if tensor.shape != expected:
        raise AssertionError(f"Expected shape {expected} but got {tuple(tensor.shape)}")
