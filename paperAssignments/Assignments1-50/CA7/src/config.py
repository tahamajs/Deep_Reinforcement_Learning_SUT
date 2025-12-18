from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """
    Training and model hyperparameters for GET-ROPR minimal reference implementation.
    These defaults are conservative and intended for smoke-testing and local development.
    """
    # Environment / data
    obs_dim: int = 8
    action_dim: int = 2
    seq_len: int = 32
    batch_size: int = 16

    # RL hyperparameters
    gamma: float = 0.99
    lam: float = 0.9
    c_rho: Optional[float] = 2.0  # IS clipping, None disables

    # Model sizes
    hidden_size: int = 128
    critic_hidden: int = 128
    actor_hidden: int = 128

    # Optimization
    lr_critic: float = 3e-4
    lr_actor: float = 3e-4
    grad_clip: float = 10.0

    # Misc
    device: str = "cpu"

