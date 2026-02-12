"""Aggregate benchmark outputs and produce frontier plots."""
from __future__ import annotations

import csv
import glob
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def read_rows(csv_files: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in csv_files:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))
    return rows


def plot_global_pareto(rows: List[Dict[str, str]], out_path: str):
    if not rows:
        return
    x = np.array([float(r["avg_cost"]) for r in rows])
    y = np.array([float(r["avg_reward"]) for r in rows])
    labels = [f"{r['algo']}:{r['scenario']}" for r in rows]

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, c="tab:blue")
    for i, lbl in enumerate(labels):
        plt.annotate(lbl, (x[i], y[i]))
    plt.xlabel("Average Cost")
    plt.ylabel("Average Reward")
    plt.title("Global Pareto Frontier")
    plt.tight_layout()
    plt.savefig(out_path)


def plot_pid_heatmap(rows: List[Dict[str, str]], out_path: str):
    pid_rows = [r for r in rows if r.get("kp") and r.get("kd")]
    if not pid_rows:
        return
    kp_vals = sorted({float(r["kp"]) for r in pid_rows})
    kd_vals = sorted({float(r["kd"]) for r in pid_rows})

    grid = np.zeros((len(kp_vals), len(kd_vals)), dtype=np.float32)
    for i, kp in enumerate(kp_vals):
        for j, kd in enumerate(kd_vals):
            candidates = [r for r in pid_rows if float(r["kp"]) == kp and float(r["kd"]) == kd]
            if not candidates:
                continue
            # frontier score: reward - 300*cost for safety-heavy ranking
            scores = [float(r["avg_reward"]) - 300.0 * float(r["avg_cost"]) for r in candidates]
            grid[i, j] = max(scores)

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    plt.figure(figsize=(6, 4))
    plt.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="Frontier score")
    plt.xticks(range(len(kd_vals)), [str(v) for v in kd_vals])
    plt.yticks(range(len(kp_vals)), [str(v) for v in kp_vals])
    plt.xlabel("kd")
    plt.ylabel("kp")
    plt.title("PID Sweep Heatmap")
    plt.tight_layout()
    plt.savefig(out_path)


def aggregate_results(results_dir: str, plots_dir: str):
    csv_files = glob.glob(os.path.join(results_dir, "**", "summary.csv"), recursive=True)
    rows = read_rows(csv_files)
    if not rows:
        return
    plot_global_pareto(rows, os.path.join(plots_dir, "global_pareto.png"))
    plot_pid_heatmap(rows, os.path.join(plots_dir, "pid_heatmap.png"))
