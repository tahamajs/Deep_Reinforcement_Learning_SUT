"""Simple experiment runner helpers for CA19.

This file provides a lightweight helper to run debug sweeps and save per-episode
metrics to CSV files. It is intentionally minimal and meant for demonstration and
reproducibility in assignments; for large experiments consider using tools like
hydra, sacred, or sacred+omniboard.
"""
from pathlib import Path
import csv
import time
from typing import Sequence

# ensure local imports work when module is loaded directly (tests use importlib.spec)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CAConfig
from train import run_training
from utils import set_seed


def run_one(cfg: CAConfig, seed: int, out_dir: Path) -> None:
    cfg.seed = seed
    set_seed(cfg.seed)
    out = run_training(cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"metrics_seed_{seed}.csv"
    # Save a short summary for the run (per-episode returns)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "step", "seed", "episode", "train_return"])
        for i, r in enumerate(out["rewards"]):
            writer.writerow([int(time.time()), out["steps"], seed, i, r])


def run_sweep(cfg: CAConfig, seeds: Sequence[int], base_out: Path) -> None:
    for s in seeds:
        run_one(cfg, s, base_out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/my_run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--total_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = CAConfig()
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps

    run_sweep(cfg, args.seeds, Path(args.out_dir))
