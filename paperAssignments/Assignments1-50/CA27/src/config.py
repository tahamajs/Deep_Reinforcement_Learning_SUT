"""Configuration dataclasses for meta-learning algorithms."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class MAMLConfig:
    """Configuration for MAML algorithm."""
    obs_dim: int
    action_dim: int
    hidden_dim: int = 64
    inner_lr: float = 0.1
    meta_lr: float = 0.001
    inner_steps: int = 1
    gamma: float = 0.99
    num_meta_iterations: int = 100
    meta_batch_size: int = 5
    num_steps_per_task: int = 50


@dataclass
class RL2Config:
    """Configuration for RL² algorithm."""
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    num_lstm_layers: int = 2
    lr: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    num_meta_iterations: int = 50
    meta_batch_size: int = 5
    num_episodes_per_task: int = 5
    ppo_epochs: int = 4
    clip_ratio: float = 0.2


@dataclass
class TaskConfig:
    """Configuration for task distribution."""
    env_name: str = "CartPole-v1"
    num_tasks: int = 50
    gravity_range: tuple[float, float] = (8.0, 11.0)
    masscart_range: tuple[float, float] = (0.8, 1.2)
    masspole_range: tuple[float, float] = (0.08, 0.12)
    length_range: tuple[float, float] = (0.4, 0.6)


@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    algorithm: str = "maml"  # "maml" or "rl2"
    maml: Optional[MAMLConfig] = None
    rl2: Optional[RL2Config] = None
    task: Optional[TaskConfig] = None
    seed: int = 42
    save_path: str = "results/"