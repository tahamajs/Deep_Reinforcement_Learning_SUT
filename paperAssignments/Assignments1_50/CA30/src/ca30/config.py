from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    seed: int = 42
    input_dim: int = 8
    hidden_dim: int = 64
    output_dim: int = 2
    epochs: int = 1
    batch_size: int = 16

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        # Validate keys and apply defaults via dataclass instantiation
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in allowed}
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in allowed}
        return cls(**data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.__dict__, f)
