"""Shared helpers for config-aware script entrypoints."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np

from projects.amasa_clean.amasa.core.config import load_config, apply_named_overlays
from projects.amasa_clean.amasa.envs import make_scenario_env, SuturingEnv
from projects.amasa_clean.amasa.offline import CQLAgent, CQLConfig, IQLAgent, IQLConfig
from projects.amasa_clean.amasa.online import SACLagrangianAgent, SACLagConfig, PPOLagrangianAgent, PPOLagConfig


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    root = project_root()
    base = root / "configs" / "base.yaml"
    config_path = Path(args.config) if getattr(args, "config", "") else base
    cfg = load_config(config_path, base_path=None if config_path == base else base)

    scenario = getattr(args, "scenario", "") or cfg["scenario"]["type"]
    algo = getattr(args, "algo", "") or cfg["algo"]["name"]
    preset = getattr(args, "preset", "") or cfg["experiment"].get("preset", "smoke")
    cfg = apply_named_overlays(
        cfg,
        root / "configs",
        scenario=scenario,
        algo=algo,
        preset=preset,
    )

    if getattr(args, "device", None):
        cfg["experiment"]["device"] = args.device
    if getattr(args, "seed", None) is not None:
        cfg["experiment"]["seed"] = args.seed
    return cfg


def make_env(cfg: Dict[str, Any], seed: int | None = None):
    scenario = cfg["scenario"]["type"]
    if scenario == "nominal":
        return SuturingEnv(max_steps=cfg["env"]["max_steps"], seed=seed)

    return make_scenario_env(
        scenario,
        max_steps=cfg["env"]["max_steps"],
        seed=seed,
        safe_force=cfg["env"]["safe_force"],
        safe_corridor=cfg["env"]["safe_corridor"],
        force_noise_scale=cfg["scenario"].get("force_noise_scale", 0.1),
        dynamics_shift=cfg["scenario"].get("dynamics_shift", 0.0),
        obs_corruption_prob=cfg["scenario"].get("obs_corruption_prob", 0.0),
        delayed_obs_prob=cfg["scenario"].get("delayed_obs_prob", 0.0),
        domain_randomization=cfg["env"].get("domain_randomization", False),
    )


def make_agent(algo_name: str, obs_dim: int, act_dim: int, cfg: Dict[str, Any]):
    device = cfg["experiment"]["device"]
    a = cfg["algo"]
    if algo_name == "cql":
        return CQLAgent(
            CQLConfig(
                obs_dim=obs_dim,
                act_dim=act_dim,
                device=device,
                discount=a.get("discount", 0.99),
                tau=a.get("tau", 0.005),
                actor_lr=a.get("lr", 3e-4),
                critic_lr=a.get("lr", 3e-4),
                cql_alpha=a.get("cql_alpha", 5.0),
            )
        )
    if algo_name == "iql":
        return IQLAgent(
            IQLConfig(
                obs_dim=obs_dim,
                act_dim=act_dim,
                device=device,
                discount=a.get("discount", 0.99),
                tau=a.get("tau", 0.005),
                lr=a.get("lr", 3e-4),
                expectile=a.get("expectile", 0.7),
                beta=a.get("beta", 3.0),
            )
        )
    if algo_name == "sac_lag":
        return SACLagrangianAgent(
            SACLagConfig(
                obs_dim=obs_dim,
                act_dim=act_dim,
                device=device,
                discount=a.get("discount", 0.99),
                tau=a.get("tau", 0.005),
                lr=a.get("lr", 3e-4),
                alpha=a.get("alpha", 0.2),
            )
        )
    if algo_name == "ppo_lag":
        return PPOLagrangianAgent(
            PPOLagConfig(
                obs_dim=obs_dim,
                act_dim=act_dim,
                device=device,
                discount=a.get("discount", 0.99),
                lr=a.get("lr", 3e-4),
            )
        )
    raise ValueError(f"Unknown algo '{algo_name}'")


def add_common_config_flags(parser: argparse.ArgumentParser):
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--preset", type=str, choices=["smoke", "full"], default="")
    parser.add_argument("--scenario", type=str, choices=["nominal", "perturbed", "adversarial"], default="")
    parser.add_argument("--algo", type=str, choices=["cql", "iql", "sac_lag", "ppo_lag"], default="")


def ensure_out_dirs(root: str):
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(root, "plots"), exist_ok=True)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}
