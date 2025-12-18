import os
import random
from pathlib import Path
import numpy as np


def set_seed(seed: int) -> None:
    """Set seeds for deterministic runs (where possible)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is optional. If not present, the numpy seed is sufficient for the tests.
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
