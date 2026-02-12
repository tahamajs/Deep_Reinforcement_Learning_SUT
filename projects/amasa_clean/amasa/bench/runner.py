"""Benchmark runner utilities for matrix execution and PID sweeps."""
from __future__ import annotations

import copy
from typing import Dict, Any, List
from pathlib import Path

from projects.amasa_clean.amasa.core.config import apply_named_overlays


def _config_root() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"


def build_benchmark_jobs(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    jobs = []
    seeds = base_cfg["eval"].get("seeds", [0, 1, 2])
    scenarios = ["nominal", "perturbed", "adversarial"]

    for algo in ["cql", "iql"]:
        for scenario in ["nominal", "perturbed"]:
            for seed in seeds:
                cfg = copy.deepcopy(base_cfg)
                cfg = apply_named_overlays(
                    cfg,
                    _config_root(),
                    algo=algo,
                    scenario=scenario,
                    preset=cfg["experiment"].get("preset", "smoke"),
                )
                cfg["experiment"]["seed"] = int(seed)
                cfg["experiment"]["mode"] = "offline_train"
                jobs.append(cfg)

    for algo in ["sac_lag", "ppo_lag"]:
        for scenario in scenarios:
            for seed in seeds:
                cfg = copy.deepcopy(base_cfg)
                cfg = apply_named_overlays(
                    cfg,
                    _config_root(),
                    algo=algo,
                    scenario=scenario,
                    preset=cfg["experiment"].get("preset", "smoke"),
                )
                cfg["experiment"]["seed"] = int(seed)
                cfg["experiment"]["mode"] = "online_train"
                jobs.append(cfg)

    return jobs


def build_pid_sweep_jobs(base_cfg: Dict[str, Any], kp_vals, kd_vals, ki=0.08) -> List[Dict[str, Any]]:
    jobs = []
    for kp in kp_vals:
        for kd in kd_vals:
            cfg = copy.deepcopy(base_cfg)
            cfg["safety"]["kp"] = float(kp)
            cfg["safety"]["kd"] = float(kd)
            cfg["safety"]["ki"] = float(ki)
            cfg["experiment"]["name"] = f"pid_kp{kp}_kd{kd}"
            cfg["experiment"]["mode"] = "online_train"
            jobs.append(cfg)
    return jobs
