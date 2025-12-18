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
    hidden_sizes: tuple[int, ...] = (64, 64)
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 50

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        """Create a Config from a plain mapping.

        Normalizes common types (e.g., list -> tuple for `hidden_sizes`) and
        validates minimal field types so downstream code can rely on stable
        types.
        """
        data = dict(d)
        # normalize hidden_sizes to tuple[int, ...]
        if "hidden_sizes" in data:
            hs = data["hidden_sizes"]
            if isinstance(hs, list):
                data["hidden_sizes"] = tuple(int(x) for x in hs)
            elif isinstance(hs, tuple):
                data["hidden_sizes"] = tuple(int(x) for x in hs)
            else:
                raise TypeError("hidden_sizes must be a list or tuple of ints")
        return Config(**data)

    @staticmethod
    def load_yaml(path: Path | str) -> "Config":
        """Load a YAML config file and return a `Config` instance."""
        p = Path(path)
        with p.open("r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("Config yaml must contain a mapping at the top level")
        return Config.from_dict(data)















