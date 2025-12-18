from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Any, Dict


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "auto"  # "cpu", "cuda", or "auto"
    input_dim: int = 16
    hidden_dims: tuple[int, ...] = (64, 64)
    output_dim: int = 1
    task: str = "regression"  # or "classification"
    save_dir: str = "outputs"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainConfig":
        # allow hidden_dims to be list in yaml
        if "hidden_dims" in d and isinstance(d["hidden_dims"], list):
            d = d.copy()
            d["hidden_dims"] = tuple(d["hidden_dims"])
        return TrainConfig(**d)


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    # allow top-level "train" section or direct fields
    if "train" in raw and isinstance(raw["train"], dict):
        raw = raw["train"]
    return TrainConfig.from_dict(raw)


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f)
