from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(RuntimeError):
    pass


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {p}")
    return data


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(default_path: str | Path, override_path: str | Path | None, cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_yaml(default_path)
    if override_path:
        cfg = deep_update(cfg, load_yaml(override_path))
    clean_cli = {k: v for k, v in cli_overrides.items() if v is not None}
    return deep_update(cfg, clean_cli)
