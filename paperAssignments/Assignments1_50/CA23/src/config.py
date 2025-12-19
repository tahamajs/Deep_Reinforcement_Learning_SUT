from dataclasses import dataclass, field, fields
from typing import List, Any, Dict, Union
from pathlib import Path
import yaml
import logging


logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Experiment hyperparameters and settings.

    This dataclass is intentionally simple and serializable. Use
    `ExperimentConfig.from_yaml(path)` to load from a YAML file. A
    `to_yaml(path)` helper is provided for convenience in tests and scripts.
    Validation is performed in ``__post_init__`` to fail fast on bad inputs.
    """

    env_name: str = "CartPole-v1"
    seed: int = 0
    device: str = "cpu"
    learning_rate: float = 1e-3
    gamma: float = 0.99
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    batch_size: int = 32
    max_episodes: int = 1000
    entropy_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not all(isinstance(n, int) and n > 0 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must be a list of positive integers")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError("Config YAML must contain a mapping at top level")
        # Only pass recognized fields to the dataclass
        known = {f.name for f in fields(cls)}
        filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in known}
        # Helpful logging for ignored fields
        ignored = set(data.keys()) - set(filtered.keys())
        if ignored:
            logger.debug("Ignoring unknown config keys: %s", sorted(ignored))
        return cls(**filtered)

    def to_yaml(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.safe_dump({fld.name: getattr(self, fld.name) for fld in fields(self)}, fh)

