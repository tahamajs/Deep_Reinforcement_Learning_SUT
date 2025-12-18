from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Environment / data
    env_name: str = "hopper-medium-expert-v2"
    seed: int = 0

    # Distributional critic
    n_quantiles: int = 50
    n_critics: int = 2
    kappa: float = 1.0  # Huber threshold

    # CVaR
    alpha_cvar: float = 0.1

    # Optimization
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    target_tau: float = 0.005
    entropy_beta: float = 0.2

    # SCAS
    lambda_base: float = 1.0
    use_adaptive_lambda: bool = True

    # Misc
    device: str = "cpu"
    max_steps: int = 1_000_000
    save_every: int = 100_000









