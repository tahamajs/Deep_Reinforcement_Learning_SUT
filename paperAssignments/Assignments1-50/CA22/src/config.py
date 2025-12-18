from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cpu"
    obs_dim: int = 8
    action_dim: int = 4
    hidden_size: int = 64
    lr: float = 3e-4
    batch_size: int = 32
    epochs: int = 50
    gamma: float = 0.99
    constraint_threshold: float = 0.5
    lagrange_lr: float = 1e-2
    max_mu: float = 1e3

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if yaml is not None:
            with p.open("r") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            # very small fallback parser for simple key: value YAML-like files
            cfg = {}
            with p.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if ":" in line:
                        k, v = line.split(":", 1)
                        cfg[k.strip()] = _simple_parse(v.strip())
        return cls(**{**_dataclass_defaults(cls), **cfg})


def _simple_parse(val: str) -> Any:
    # try int, float, bool, else string
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    try:
        return int(val)
    except Exception:
        pass
    try:
        return float(val)
    except Exception:
        pass
    return val


def _dataclass_defaults(dc) -> Dict[str, Any]:
    return {f.name: f.default for f in dc.__dataclass_fields__.values()}








