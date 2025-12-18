"""
Load YAML configs into cfg for CA8.
This module is import-safe and has no heavy deps besides PyYAML.
"""

from typing import Any, Dict
from pathlib import Path

import yaml

from config import cfg  # type: ignore


def load_config_from_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file and set attributes on cfg where keys match.
    Returns the parsed dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping")
    for k, v in data.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)  # type: ignore[attr-defined]
            except Exception:
                # cfg is frozen dataclass; create a shallow replacement
                new_vals = cfg.as_dict()
                new_vals[k] = v
                # recreate cfg
                from config import Config  # type: ignore

                # replace module-level cfg variable
                globals()["cfg"] = Config(**new_vals)  # type: ignore
        else:
            # ignore unknown keys
            pass
    return data






