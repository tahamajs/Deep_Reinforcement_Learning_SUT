from typing import Any, Dict
import random
import os
import torch
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str, device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device)








