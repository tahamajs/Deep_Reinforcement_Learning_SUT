from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    input_dim: int = 1
    hidden_dims: tuple[int, ...] = (64, 64)
    output_dim: int = 1
    activation: str = "relu"


@dataclass
class TrainConfig:
    seed: int = 0
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50
    device: str = "cpu"


@dataclass
class ExperimentConfig:
    name: str = "ca26_experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: Path | str) -> ExperimentConfig:
    """Load YAML config from file and return an ExperimentConfig.

    This keeps the code import-safe and small; if `yaml` is not available the
    function will raise a descriptive error.
    """
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh)
    # Map dict -> dataclasses conservatively
    model_raw: Dict[str, Any] = raw.get("model", {})
    train_raw: Dict[str, Any] = raw.get("train", {})
    model_cfg = ModelConfig(**model_raw)
    train_cfg = TrainConfig(**train_raw)
    name = raw.get("name", "ca26_experiment")
    return ExperimentConfig(name=name, model=model_cfg, train=train_cfg)
