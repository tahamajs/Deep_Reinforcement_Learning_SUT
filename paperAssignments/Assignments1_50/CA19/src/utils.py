from typing import Any, Dict
import random
import os
import shutil
import tempfile
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Set seeds for python, numpy and torch to improve reproducibility.

    This function also configures cuDNN to deterministic mode when CUDA is
    available to reduce nondeterminism between runs. Note: fully deterministic
    behaviour across platforms and PyTorch versions is not guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # make CUDA deterministic where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a batch (mapping of keys to tensors) to the given device."""
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Atomically save checkpoint to `path`.

    Uses a temporary file in the same directory and moves it into place to avoid
    partial writes if the process is interrupted.
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    # atomic save via temporary file
    fd, tmp = tempfile.mkstemp(dir=d if d else None)
    os.close(fd)
    try:
        torch.save(state, tmp)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_checkpoint(
    path: str, device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """Load checkpoint from path and map tensors to `device`.

    Raises FileNotFoundError if path does not exist. The function conservatively
    returns the dict saved by `save_checkpoint`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)















