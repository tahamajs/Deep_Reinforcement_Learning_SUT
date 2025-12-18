from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Config:
    """Configuration container for CA17 experiments.

    Keep all hyperparameters here so other modules import this file
    and remain import-safe.
    """

    env_name: str = "CartPole-v1"
    seed: int = 42
    lr: float = 1e-3
    hidden_size: int = 128
    gamma: float = 0.99
    total_timesteps: int = 50_000
    rollout_length: int = 2048
    batch_size: int = 64
    save_dir: Path = Path("outputs/ca17")
    device: Optional[str] = None  # "cpu" or "cuda"


def get_default_config() -> Config:
    """Return a frozen default configuration instance."""
    return Config()








