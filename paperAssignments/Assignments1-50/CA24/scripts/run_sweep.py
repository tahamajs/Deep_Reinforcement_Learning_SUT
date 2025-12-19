#!/usr/bin/env python3
"""Simple sweep runner for CA24 experiments.

This script is intentionally small and uses `src` functions; adapt for your experiments.
"""
from pathlib import Path
import json
from itertools import product

from src.config import Config
from src.experiment import run_experiment


def save_metrics(out_dir: Path, run_id: str, cfg: Config, result: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, **result}, f, indent=2)


def main(outdir: str = "outputs/sweep", seeds=(42, 43, 44)):
    outdir = Path(outdir)
    hidden_options = [[64, 64], [32], [128, 128]]
    lr_options = [1e-3, 1e-4]

    for hidden, lr in product(hidden_options, lr_options):
        setting = f"hidden={hidden}_lr={lr}"
        for seed in seeds:
            cfg = Config(seed=seed, hidden_dims=hidden, lr=lr, epochs=5)
            run_id = f"{setting}/seed-{seed}"
            result = run_experiment(cfg)
            run_path = outdir / setting / f"seed-{seed}"
            save_metrics(run_path, run_id, cfg, result)
            print(f"Saved {run_id} -> {run_path}")


if __name__ == "__main__":
    main()
