import os
import json
import csv
import time
from typing import Any, Dict

import torch


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=device)


class CSVLogger:
    """
    Simple CSV logger that appends rows with a header. Meant for lightweight experiments.
    """

    def __init__(self, path: str, header: Dict[str, str]):
        self.path = path
        self.header = list(header.keys())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def log(self, row: Dict[str, Any]):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row.get(k, "") for k in self.header])

    def flush(self):
        # no-op for simple file-based logging
        return


def compute_staleness(stored_hidden: torch.Tensor, recomputed_hidden: torch.Tensor) -> torch.Tensor:
    """
    Compute per-hidden-vector staleness metric between stored and recomputed hidden states.

    Staleness is defined as 1 - cosine_similarity(stored, recomputed) and returns
    a tensor of shape [...], broadcasted over batch/time as appropriate.

    Args:
        stored_hidden: tensor of shape [B, H] or [..., H]
        recomputed_hidden: tensor of same shape as stored_hidden

    Returns:
        staleness: same leading shape as inputs (no final H dim), values in [0, 2]
    """
    # flatten final dim to compute cosine similarity
    s = stored_hidden.view(-1, stored_hidden.shape[-1])
    r = recomputed_hidden.view(-1, recomputed_hidden.shape[-1])
    s_norm = s.norm(dim=-1).clamp_min(1e-8)
    r_norm = r.norm(dim=-1).clamp_min(1e-8)
    cos = (s * r).sum(dim=-1) / (s_norm * r_norm)
    staleness = 1.0 - cos
    return staleness.view(*stored_hidden.shape[:-1])















