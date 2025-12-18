"""Utility helpers: seeding, file I/O, simple plotting helpers.

Design goals:
- Keep functions small and well-typed.
- No heavy side effects on import.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set seeds for repeatability across RNGs.

    Args:
        seed: integer seed
        deterministic: when True, enable PyTorch deterministic flags (may
            reduce performance and not be fully reproducible across platforms).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # These settings help reproducibility for many ops but are not a
        # guarantee across devices/platforms.
        torch.use_deterministic_algorithms(True)


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(device: Optional[str] = None) -> torch.device:
    """Resolve a device string to a ``torch.device``.

    Accepts None (auto), 'cpu', 'cuda', or full torch device strings like
    'cuda:0'.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def save_figure(fig: object, path: Union[str, Path], dpi: int = 300) -> None:
    """Save a matplotlib Figure or Axes to disk, ensuring the directory exists.

    The function accepts a ``matplotlib.figure.Figure`` instance, an Axes
    instance (in which case its parent figure is used), or any object with a
    ``savefig`` method.
    """
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")

    p = ensure_dir(Path(path).parent)

    # Normalize to a Figure
    if isinstance(fig, Axes):
        fig = fig.figure
    if not isinstance(fig, Figure):
        # If it has savefig, try to call it directly; otherwise raise
        if not hasattr(fig, "savefig"):
            raise TypeError("fig must be a matplotlib Figure or Axes or have savefig")
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
