from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Hyperparameters and configuration for RA-U-OBAC experiments."""

    seed: int = 42
    env_name: str = "Hopper-v2"
    # For quick local demos default to a lightweight env
    # (overridden in experiments to use MuJoCo / D4RL datasets)
    env_name: str = "CartPole-v1"
    max_steps: int = 500_000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    tau: float = 0.005
    device: str = "cpu"

    # Retrieval buffer
    buffer_size: int = 1_000_000
    trajectory_max_len: int = 1000
    retrieval_k: int = 10
    retrieval_nn: int = 50
    retrieval_delta: float = 1e9  # not used in simple L2 search, placeholder

    # Ensemble critics
    critic_ensemble_size: int = 4

    # Boosting / uncertainty
    lambda_blend: float = 0.75
    beta_uq: float = 1.0
    uq_threshold: Optional[float] = None

    # Training schedules
    actor_update_every: int = 1
    critic_updates: int = 1
    offline_actor_updates: int = 1

    # Logging / checkpoint
    log_interval: int = 1000
    save_interval: int = 50_000

    # numerical stabilization
    eps: float = 1e-8

    def device_str(self) -> str:
        return self.device
