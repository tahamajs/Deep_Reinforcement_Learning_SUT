from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    """Hyperparameters and runtime configuration for AU-DMG experiments."""

    seed: int = 42
    device: str = "cpu"
    env: str = "antmaze-medium-diverse-v2"
    ensemble_size: int = 4
    latent_dim: int = 16
    candidate_M: int = 20
    candidate_sigma: float = 0.1
    eps: float = 0.2
    beta: float = 0.35
    kappa: float = 12.0
    gamma: float = 0.99
    batch_size: int = 256
    lr: float = 3e-4
    tau: float = 0.005
    lcb_coef: float = 0.0
    use_lcb: bool = False
    max_grad_norm: float = 10.0


def default_config() -> Config:
    return Config()









