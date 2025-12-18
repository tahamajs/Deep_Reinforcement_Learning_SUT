from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class Config:
    seed: int = 42
    device: str = "cpu"
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 5
    gamma: float = 0.99
    constraint_c: float = 0.1
    lambda_lr: float = 1e-2
    lambda_clip: float = 100.0
    hidden_dim: int = 128
    obs_dim: int = 8
    action_dim: int = 2

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        return Config(**{**dataclasses.asdict(Config()), **data})

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)









