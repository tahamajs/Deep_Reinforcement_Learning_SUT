"""Aggregate CSV metric files produced by `src/experiment.py`.

Usage:
    python scripts/aggregate_results.py outputs/my_run

This script reads `metrics_seed_*.csv` files and prints simple aggregate statistics.
"""
from pathlib import Path
import csv
import math
import statistics
import sys


def aggregate(out_dir: Path):
    files = sorted(out_dir.glob("metrics_seed_*.csv"))
    if not files:
        print("No metric files found in", out_dir)
        return
    per_seed = {}
    for f in files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            returns = [float(row["train_return"]) for row in reader]
            seed = f.stem.split("_")[-1]
            per_seed[seed] = returns

    # compute mean final return per seed and overall mean/std
    finals = [s[-1] if len(s) > 0 else float("nan") for s in per_seed.values()]
    finals = [v for v in finals if not math.isnan(v)]
    if finals:
        print(f"Seeds: {len(finals)}")
        print(f"Final return mean: {statistics.mean(finals):.3f}")
        if len(finals) > 1:
            print(f"Final return std: {statistics.stdev(finals):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/aggregate_results.py <out_dir>")
        raise SystemExit(1)
    aggregate(Path(sys.argv[1]))
