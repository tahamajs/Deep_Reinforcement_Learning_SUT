import os
import json
import random
from typing import Any, Dict, Optional
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across torch, numpy and random."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    """Save a training checkpoint (PyTorch state dict + metadata)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a training checkpoint saved with save_checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")






