from dataclasses import dataclass
import torch


@dataclass
class CAConfig:
    seed: int = 0
    lr: float = 3e-4
    batch_size: int = 128
    gamma: float = 0.99
    obs_dim: int = 4
    action_dim: int = 2
    ensemble_size: int = 3
    hidden_dim: int = 64
    beta: float = 0.1
    env_name: str = "CartPole-v1"
    max_steps_per_episode: int = 500
    total_steps: int = 2000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

