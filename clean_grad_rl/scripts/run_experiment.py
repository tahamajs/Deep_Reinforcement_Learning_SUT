#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from grad_rl.algorithms import CHAIN_REGISTRY
from grad_rl.core import deep_update, load_yaml, set_seed


def _default_config_path(chain: str) -> Path:
    return Path("configs/chains") / f"{chain}.yaml"


def _as_dict(x):
    return x if isinstance(x, dict) else {}


def resolve_experiment_config(chain: str, algo: str, config_path: str | None) -> Dict[str, Any]:
    cfg_file = Path(config_path) if config_path else _default_config_path(chain)
    root = load_yaml(cfg_file)
    common = _as_dict(root.get("common", {}))
    algo_cfg = _as_dict(_as_dict(root.get("algorithms", {})).get(algo, {}))
    return deep_update(common, algo_cfg)


def parse_args():
    p = argparse.ArgumentParser(description="Unified DRL experiment runner")
    p.add_argument("--chain", required=True, choices=["value", "policy", "actor_critic", "model_based", "marl"])
    p.add_argument("--algo", required=True)
    p.add_argument("--env", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def run_one(chain: str, algo: str, cfg: Dict[str, Any], seed: int, out_dir: Path):
    if chain not in CHAIN_REGISTRY:
        raise ValueError(f"Unknown chain: {chain}")
    if algo not in CHAIN_REGISTRY[chain]:
        raise ValueError(f"Unknown algo '{algo}' for chain '{chain}'. Available: {list(CHAIN_REGISTRY[chain].keys())}")
    train_fn = CHAIN_REGISTRY[chain][algo]
    return train_fn(cfg, out_dir, seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = resolve_experiment_config(args.chain, args.algo, args.config)
    if args.env is not None:
        cfg["env"] = args.env
    if args.steps is not None:
        cfg["total_steps"] = args.steps
    if args.episodes is not None:
        cfg["episodes"] = args.episodes

    env_name = str(cfg.get("env", "unknown")).replace("/", "_")
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/runs") / args.chain / args.algo / env_name / f"seed_{args.seed}"

    payload = run_one(args.chain, args.algo, cfg, args.seed, out_dir)
    print(json.dumps({
        "status": "ok",
        "chain": args.chain,
        "algo": args.algo,
        "env": cfg.get("env"),
        "seed": args.seed,
        "out_dir": str(out_dir),
        "eval": payload.get("eval", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
