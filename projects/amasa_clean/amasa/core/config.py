"""YAML-first config loading with schema validation and preset application."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_TOP_LEVEL = {
    "experiment",
    "env",
    "scenario",
    "algo",
    "safety",
    "train",
    "eval",
}


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def load_config(path: str | Path, base_path: str | Path | None = None) -> Dict[str, Any]:
    cfg = _load_yaml(path)
    if base_path is not None:
        base_cfg = _load_yaml(base_path)
        cfg = merge_dicts(base_cfg, cfg)
    validate_config(cfg)
    return cfg


def apply_preset(cfg: Dict[str, Any], preset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = merge_dicts(cfg, preset_cfg)
    validate_config(merged)
    return merged


def validate_config(cfg: Dict[str, Any]):
    missing = REQUIRED_TOP_LEVEL - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing sections: {sorted(missing)}")

    scenario_type = cfg["scenario"].get("type")
    if scenario_type not in {"nominal", "perturbed", "adversarial"}:
        raise ValueError("scenario.type must be one of nominal/perturbed/adversarial")

    preset = cfg["experiment"].get("preset", "smoke")
    if preset not in {"smoke", "full"}:
        raise ValueError("experiment.preset must be smoke or full")

    algo_name = cfg["algo"].get("name")
    if algo_name not in {"cql", "iql", "sac_lag", "ppo_lag"}:
        raise ValueError("algo.name must be cql/iql/sac_lag/ppo_lag")

    for key in ("kp", "ki", "kd"):
        if key not in cfg["safety"]:
            raise ValueError(f"safety.{key} is required")
