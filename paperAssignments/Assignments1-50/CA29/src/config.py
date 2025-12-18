"""Configuration management for SAC experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class SACConfig:
    """Configuration for Soft Actor-Critic (SAC) algorithm.

    Attributes:
        env_name: Name of the Gym environment (e.g., 'HalfCheetah-v4').
        gamma: Discount factor for future rewards.
        alpha: Initial temperature parameter for entropy regularization.
        lr_actor: Learning rate for the actor (policy) network.
        lr_critic: Learning rate for the critic (Q-function) networks.
        buffer_size: Size of the replay buffer.
        batch_size: Batch size for training updates.
        num_steps: Total number of environment steps for training.
        eval_freq: Frequency of evaluation episodes.
        seed: Random seed for reproducibility.
        device: Device to run on ('cpu' or 'cuda').
        log_dir: Directory to save logs and models.
    """
    env_name: str = "HalfCheetah-v4"
    gamma: float = 0.99
    alpha: float = 0.2
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    num_steps: int = 1_000_000
    eval_freq: int = 10_000
    seed: int = 42
    device: str = "auto"  # 'auto', 'cpu', or 'cuda'
    log_dir: str = "results/sac_experiment"


def load_config(config_path: str) -> SACConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        SACConfig instance with loaded parameters.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If the config file does not exist.
    """
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML configs. Install with: pip install PyYAML")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return SACConfig(**data)


def save_config(config: SACConfig, config_path: str) -> None:
    """Save configuration to a YAML file.

    Args:
        config: SACConfig instance to save.
        config_path: Path to save the YAML file.
    """
    if yaml is None:
        raise ImportError("PyYAML is required to save YAML configs. Install with: pip install PyYAML")

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config.__dict__, f)