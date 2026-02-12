"""Aggregate benchmark outputs and produce frontier/diagnostic plots."""
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


def _maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _frontier_score(row: Dict[str, str]) -> float:
    reward = _maybe_float(row.get("avg_reward"))
    cost = _maybe_float(row.get("avg_cost"))
    if reward is None or cost is None:
        return -1e9
    return reward - 300.0 * cost


def plot_global_pareto(rows: List[Dict[str, str]], out_path: str):
    if not rows:
        return
    valid = [r for r in rows if _maybe_float(r.get("avg_cost")) is not None and _maybe_float(r.get("avg_reward")) is not None]
    if not valid:
        return

    x = np.array([float(r["avg_cost"]) for r in valid])
    y = np.array([float(r["avg_reward"]) for r in valid])
    labels = [f"{r.get('algo', 'na')}:{r.get('scenario', 'na')}" for r in valid]
    scenarios = [r.get("scenario", "na") for r in valid]
    cmap = {"nominal": "tab:blue", "perturbed": "tab:orange", "adversarial": "tab:red"}
    colors = [cmap.get(s, "tab:gray") for s in scenarios]

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, c=colors, alpha=0.85)
    for i, lbl in enumerate(labels):
        plt.annotate(lbl, (x[i], y[i]))
    plt.xlabel("Average Cost")
    plt.ylabel("Average Reward")
    plt.title("Global Pareto Frontier")
    plt.tight_layout()
    plt.savefig(out_path)


def plot_scenario_frontiers(rows: List[Dict[str, str]], out_dir: str):
    scenarios = ["nominal", "perturbed", "adversarial"]
    os.makedirs(out_dir, exist_ok=True)
    for scenario in scenarios:
        subset = [r for r in rows if r.get("scenario") == scenario and _maybe_float(r.get("avg_cost")) is not None and _maybe_float(r.get("avg_reward")) is not None]
        if not subset:
            continue
        subset = sorted(subset, key=lambda r: float(r["avg_cost"]))
        x = np.array([float(r["avg_cost"]) for r in subset], dtype=np.float32)
        y = np.array([float(r["avg_reward"]) for r in subset], dtype=np.float32)
        labels = [f"{r.get('algo', 'na')}:s{r.get('seed', '0')}" for r in subset]

        plt.figure(figsize=(6, 4))
        plt.plot(x, y, "-o", color="tab:blue", linewidth=1.2, markersize=4)
        for i, lbl in enumerate(labels):
            plt.annotate(lbl, (x[i], y[i]))
        plt.xlabel("Average Cost")
        plt.ylabel("Average Reward")
        plt.title(f"Reward-Cost Curve: {scenario}")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pareto_{scenario}.png"))


def plot_pid_heatmaps(rows: List[Dict[str, str]], out_dir: str):
    pid_rows = [r for r in rows if r.get("kp") and r.get("kd")]
    if not pid_rows:
        return
    kp_vals = sorted({float(r["kp"]) for r in pid_rows})
    kd_vals = sorted({float(r["kd"]) for r in pid_rows})

    reward_grid = np.full((len(kp_vals), len(kd_vals)), np.nan, dtype=np.float32)
    cost_grid = np.full((len(kp_vals), len(kd_vals)), np.nan, dtype=np.float32)
    frontier_grid = np.full((len(kp_vals), len(kd_vals)), np.nan, dtype=np.float32)
    for i, kp in enumerate(kp_vals):
        for j, kd in enumerate(kd_vals):
            candidates = [r for r in pid_rows if float(r["kp"]) == kp and float(r["kd"]) == kd]
            if not candidates:
                continue
            reward_grid[i, j] = max(float(r["avg_reward"]) for r in candidates)
            cost_grid[i, j] = min(float(r["avg_cost"]) for r in candidates)
            scores = [_frontier_score(r) for r in candidates]
            frontier_grid[i, j] = max(scores)

    os.makedirs(out_dir, exist_ok=True)

    def _draw(grid: np.ndarray, title: str, label: str, path: str, cmap: str):
        plt.figure(figsize=(6, 4))
        plt.imshow(grid, origin="lower", cmap=cmap, aspect="auto")
        plt.colorbar(label=label)
        plt.xticks(range(len(kd_vals)), [str(v) for v in kd_vals])
        plt.yticks(range(len(kp_vals)), [str(v) for v in kp_vals])
        plt.xlabel("kd")
        plt.ylabel("kp")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)

    _draw(reward_grid, "PID Sweep Heatmap (Reward)", "Best reward", os.path.join(out_dir, "pid_heatmap_reward.png"), "viridis")
    _draw(cost_grid, "PID Sweep Heatmap (Cost)", "Best (min) cost", os.path.join(out_dir, "pid_heatmap_cost.png"), "magma_r")
    frontier_path = os.path.join(out_dir, "pid_heatmap_frontier.png")
    _draw(frontier_grid, "PID Sweep Heatmap (Frontier)", "Reward - 300*Cost", frontier_path, "plasma")
    # Backward-compatible filename for earlier scripts/reports.
    _draw(frontier_grid, "PID Sweep Heatmap", "Reward - 300*Cost", os.path.join(out_dir, "pid_heatmap.png"), "plasma")


def plot_best_safety_timeline(results_dir: str, out_path: str):
    summary_files = glob.glob(os.path.join(results_dir, "**", "summary.csv"), recursive=True)
    rows = read_rows(summary_files)
    online_rows = [r for r in rows if r.get("algo") in {"sac_lag", "ppo_lag"} and _maybe_float(r.get("avg_reward")) is not None]
    if not online_rows:
        return
    best = max(online_rows, key=_frontier_score)
    algo = best.get("algo", "")
    scenario = best.get("scenario", "")
    seed = str(best.get("seed", ""))

    timeline_files = glob.glob(os.path.join(results_dir, "**", "safety_timeline.csv"), recursive=True)
    if not timeline_files:
        return

    selected = None
    for path in timeline_files:
        p = path.replace("\\", "/")
        if algo and algo in p and scenario in p and f"seed{seed}" in p:
            selected = path
            break
    if selected is None:
        selected = timeline_files[0]

    with open(selected, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        timeline = list(reader)
    if not timeline:
        return

    steps = np.array([int(row["step"]) for row in timeline], dtype=np.int32)
    lambdas = np.array([float(row["lambda_value"]) for row in timeline], dtype=np.float32)
    risks = np.array([float(row["risk_score"]) for row in timeline], dtype=np.float32)
    blocked = np.array([float(row["shield_blocked"]) for row in timeline], dtype=np.float32)
    window = min(100, max(10, len(timeline) // 20))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    block_rate = np.convolve(blocked, kernel, mode="same")

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(steps, lambdas, label="lambda", color="tab:red")
    axes[0].plot(steps, risks, label="risk_score", color="tab:blue")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.2)

    axes[1].plot(steps, block_rate, label="block_rate(ma)", color="tab:green")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Safety Event Timeline (Best Frontier Run)")
    fig.tight_layout()
    fig.savefig(out_path)


def aggregate_results(results_dir: str, plots_dir: str):
    csv_files = glob.glob(os.path.join(results_dir, "**", "summary.csv"), recursive=True)
    rows = read_rows(csv_files)
    if not rows:
        return
    plot_global_pareto(rows, os.path.join(plots_dir, "global_pareto.png"))
    plot_scenario_frontiers(rows, plots_dir)
    plot_pid_heatmaps(rows, plots_dir)
    plot_best_safety_timeline(results_dir, os.path.join(plots_dir, "safety_timeline_best.png"))
