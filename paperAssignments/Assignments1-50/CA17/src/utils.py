from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np


def set_seed(seed: int) -> None:
    """Make runs deterministic by seeding stdlib, numpy, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch not installed in the environment yet; that's fine for static analysis
        pass


def ensure_dir(path: Optional[Path]) -> Path:
    p = Path(path or "outputs")
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(state: dict, path: os.PathLike) -> None:
    """Save a model checkpoint using torch.save if torch is available."""
    try:
        import torch

        torch.save(state, path)
    except Exception:
        # Fallback to a numpy save for dictionaries of arrays
        import numpy as _np

        _np.save(path, state)


def load_checkpoint(path: os.PathLike):
    try:
        import torch

        return torch.load(path, map_location="cpu")
    except Exception:
        import numpy as _np

        return _np.load(path, allow_pickle=True).item()


