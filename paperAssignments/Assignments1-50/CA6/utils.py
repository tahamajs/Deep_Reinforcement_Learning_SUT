import random
from typing import Iterable, Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]) -> None:
    """
    Set random seeds for reproducibility.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if len(xs) == 0:
        return 0.0
    return float(sum(xs) / len(xs))











