"""Utility functions for seeding and environment setup."""
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: Optional[int] = None) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed. If None, uses a random seed.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDA behavior deterministic where possible
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            # Some minimal PyTorch builds may not expose cudnn settings; ignore in that case
            pass