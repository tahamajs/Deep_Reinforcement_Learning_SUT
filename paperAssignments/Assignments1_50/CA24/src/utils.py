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
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # If torch is not available (e.g., during lightweight static checks), skip CUDA seeding.
        pass


def get_device(prefer_gpu: bool = False):
    """Return a device-like object based on availability and preference.

    When `torch` is unavailable, returns a string fallback ('cpu' or 'cuda').
    """
    try:
        import torch

        if prefer_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except Exception:
        return "cpu" if not prefer_gpu else "cuda"


def cpu_float(t) -> float:
    """Return a Python float from a tensor-like object.

    Works with torch tensors or numeric scalars.
    """
    try:
        return float(t.detach().cpu())
    except Exception:
        return float(t)
