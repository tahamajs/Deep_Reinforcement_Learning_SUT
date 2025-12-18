from __future__ import annotations

import random
import os
from typing import Optional

import torch


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility.

    This affects Python random, os level, and torch (both CPU and CUDA).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = False) -> torch.device:
    """Return a torch.device based on availability and preference.

    Returns 'cpu' if CUDA is unavailable or prefer_gpu is False.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cpu_float(t: torch.Tensor) -> float:
    return float(t.detach().cpu())
