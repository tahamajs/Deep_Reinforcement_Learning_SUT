import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(x: Any, device: str = "cpu") -> torch.Tensor:
    """Convert numpy/sequence to torch tensor on device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x).to(device)


def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters())















