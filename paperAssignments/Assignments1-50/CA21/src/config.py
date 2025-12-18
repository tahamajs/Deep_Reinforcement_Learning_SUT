from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(frozen=True)
class Config:
    """
    Centralized hyperparameters for CA21 demo.

    All training scripts and notebooks should import and use this Config.
    """

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim: int = 8
    hidden_dim: int = 64
    action_dim: int = 4
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10


def get_default_config() -> Config:
    """Return the default Config instance."""
    return Config()


