from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class Config:
    seed: int = 42
    lr: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    device: str = "cpu"
    hidden_sizes: tuple = (64, 64)
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 50

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        return Config(**d)

    @staticmethod
    def load_yaml(path: Path) -> "Config":
        with path.open("r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("Config yaml must contain a mapping at the top level")
        return Config.from_dict(data)

