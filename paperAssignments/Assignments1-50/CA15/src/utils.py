from typing import Any, Dict
import random
import os
import json

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str, model: torch.nn.Module, optimizer: Any, extra: Dict[str, Any] = None
) -> None:
    """Save model and optimizer state to `path` (atomic)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": getattr(optimizer, "state_dict", lambda: {})(),
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str, model: torch.nn.Module, optimizer: Any = None
) -> Dict[str, Any]:
    """Load checkpoint and restore model/optimizer states if available."""
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload.get("model_state", {}))
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload.get("extra", {})






