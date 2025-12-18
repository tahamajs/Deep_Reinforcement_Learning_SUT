from dataclasses import dataclass
from typing import Sequence
import torch


@dataclass
class Config:
    """Configuration and hyperparameters for CA3 assignment.

    All hyperparameters are centralized here so notebooks and scripts import
    a single source of truth.
    """

    # Environment
    env_name: str = "CartPole-v1"

    # Seed and device
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    lr: float = 1e-3
    gamma: float = 0.99
    max_episodes: int = 1000
    max_steps_per_episode: int = 500
    batch_size: int = 64

    # Model
    hidden_sizes: Sequence[int] = (64, 64)

    # Logging / checkpoints
    save_dir: str = "./outputs/ca3"
    checkpoint_every: int = 100

    # Misc
    max_grad_norm: float = 0.5

    def to_dict(self):
        """Return a plain dict copy of the config."""
        return {
            "env_name": self.env_name,
            "seed": self.seed,
            "device": self.device,
            "lr": self.lr,
            "gamma": self.gamma,
            "max_episodes": self.max_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "batch_size": self.batch_size,
            "hidden_sizes": tuple(self.hidden_sizes),
            "save_dir": self.save_dir,
            "checkpoint_every": self.checkpoint_every,
            "max_grad_norm": self.max_grad_norm,
        }
