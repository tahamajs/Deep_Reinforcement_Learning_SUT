"""Simple plotting utilities for planner analysis (uses matplotlib)."""

from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from typing import List


def load_metrics(csv_path: str) -> List[dict]:
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items()})
    return rows


def plot_success_vs_steps(metrics: List[dict], out: str):
    steps = [m["step"] for m in metrics]
    succ = [m.get("success_rate", 0.0) for m in metrics]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, succ, label="success_rate")
    plt.xlabel("steps")
    plt.ylabel("success_rate")
    plt.grid(True)
    plt.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: plot_planner_effect.py metrics.csv out.png")
    else:
        rows = load_metrics(sys.argv[1])
        plot_success_vs_steps(rows, sys.argv[2])
