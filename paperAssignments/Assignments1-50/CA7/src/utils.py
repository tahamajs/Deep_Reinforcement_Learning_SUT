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












