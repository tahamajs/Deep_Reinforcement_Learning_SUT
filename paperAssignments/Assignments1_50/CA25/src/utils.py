from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for python, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Resolve device string to a torch.device. "auto" chooses cuda if available."""
    prefer = (prefer or "auto").lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
