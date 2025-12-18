from dataclasses import dataclass
from typing import Any, Dict
import yaml


@dataclass
class Config:
    """Simple experiment configuration for CA15."""

    seed: int = 0
    input_dim: int = 8
    hidden_dim: int = 64
    output_dim: int = 4
    lr: float = 1e-3
    device: str = "cpu"
    batch_size: int = 32
    epochs: int = 10

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        cfg = Config()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return Config.from_dict(data or {})











