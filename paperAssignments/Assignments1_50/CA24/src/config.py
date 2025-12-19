from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Configuration dataclass for CA24 experiments.

    Keep defaults minimal and provide parsing helpers.
    """

    seed: int = 42
    device: str = "cpu"  # 'cpu' or 'cuda'
    input_dim: int = 10
    output_dim: int = 1
    hidden_dims: List[int] = (64, 64)
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10


def load_from_yaml(path: str) -> Config:
    """Load config from a YAML file.

    Importing `yaml` is deferred to avoid making `src` heavy at import time.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(**raw)
