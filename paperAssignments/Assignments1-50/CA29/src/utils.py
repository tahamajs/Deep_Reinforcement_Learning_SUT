"""Utility functions for seeding, device handling, and deterministic setup."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_str: str = "auto") -> torch.device:
    """Get the appropriate device for computation.

    Args:
        device_str: 'auto', 'cpu', or 'cuda'. 'auto' selects CUDA if available.

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str in ["cpu", "cuda"]:
        return torch.device(device_str)
    else:
        raise ValueError(f"Invalid device string: {device_str}. Use 'auto', 'cpu', or 'cuda'.")


def make_deterministic() -> None:
    """Enable deterministic mode for PyTorch operations.

    This may reduce performance but ensures reproducibility.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: torch.use_deterministic_algorithms(True) can be added for PyTorch >= 1.8
    # but may not be available in all environments.


def set_env_seed(env, seed: int) -> None:
    """Set seed for a Gym environment if it supports seeding.

    Args:
        env: The Gym environment instance.
        seed: The seed value.
    """
    try:
        env.seed(seed)
    except AttributeError:
        # For newer Gym versions, use env.reset(seed=seed)
        pass