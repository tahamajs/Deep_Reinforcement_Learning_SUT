import os
import json
import random
from pathlib import Path
import numpy as np
from typing import Any


def make_rng(seed: int | None = None) -> np.random.RandomState:
    """Create a numpy RandomState instance for deterministic sampling."""
    if seed is None:
        return np.random.RandomState()
    return np.random.RandomState(int(seed))


def set_seed(seed: int | None) -> None:
    """Set seeds for deterministic runs (where possible)."""
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        # Torch optional; ignore when not present
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open('r') as f:
        return json.load(f)
