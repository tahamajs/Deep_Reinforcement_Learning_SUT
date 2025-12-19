from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
# Use a non-interactive backend so plotting works in headless CI/environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across numpy and PyTorch."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_loss_curve(losses: Iterable[float], out: Path | str) -> None:
    out = ensure_dir(Path(out).parent)
    fig, ax = plt.subplots()
    ax.plot(list(losses), marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    fig.tight_layout()
    fig.savefig(str(out / "loss_curve.png"), dpi=150)
    plt.close(fig)


@dataclass
class FitResult:
    losses: list[float]
    final_state: dict | None = None
