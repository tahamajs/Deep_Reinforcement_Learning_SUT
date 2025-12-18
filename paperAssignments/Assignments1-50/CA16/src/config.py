from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for CA16 simple RL models and utilities."""

    env_name: str = "CartPole-v1"
    seed: int = 42
    lr: float = 3e-4
    gamma: float = 0.99
    hidden_dim: int = 128
    obs_dim: Optional[int] = None
    action_dim: Optional[int] = None
    device: Optional[str] = None
    batch_size: int = 64


def get_default_config() -> Config:
    return Config()













