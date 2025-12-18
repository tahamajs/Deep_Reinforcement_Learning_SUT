"""
Configuration and hyperparameters for CA8: MaxSink
This file centralizes defaults. Import-safe.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Config:
    # Environment / training
    env_name: str = "MiniGrid-Empty-8x8-v0"
    seed: int = 42
    total_steps: int = 200_000
    batch_size: int = 256
    start_updates: int = 1_000
    update_every: int = 1

    # Agent / distributional
    gamma: float = 0.99
    particles: int = 32
    particle_dim: int = 1  # scalar returns
    tau: float = 0.005
    lr: float = 3e-4
    grad_clip: float = 10.0

    # Sinkhorn / OT
    sinkhorn_blur: float = 0.01
    sinkhorn_scaling: float = 0.9
    sinkhorn_p: int = 2

    # Reward transform
    beta: float = 0.4  # progress bonus

    # Misc
    device: str = "cpu"

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dict__.keys()}


cfg = Config()
