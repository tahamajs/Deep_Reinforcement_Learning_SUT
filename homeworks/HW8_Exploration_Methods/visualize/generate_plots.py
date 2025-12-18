#!/usr/bin/env python3
"""
Generate plots from bandit results saved as a numpy .npz archive.

Usage:
    python generate_plots.py --results ../results/bandit_results.npz --out_dir ../pictures
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str):
    data = np.load(path)
    # Group names by suffix
    names = set(k.rsplit("_", 1)[0] for k in data.files)
    results = {}
    for name in names:
        rewards_key = f"{name}_rewards"
        regrets_key = f"{name}_regrets"
        rewards = data[rewards_key] if rewards_key in data.files else None
        regrets = data[regrets_key] if regrets_key in data.files else None
        results[name] = {"rewards": rewards, "regrets": regrets}
    return results


def plot_rewards(results: dict, out_path: str):
    plt.figure(figsize=(8, 4.5))
    for name, d in results.items():
        if d["rewards"] is None:
            continue
        plt.plot(d["rewards"], label=name)
    plt.xlabel("Timestep")
    plt.ylabel("Average reward")
    plt.title("Average reward over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_regrets(results: dict, out_path: str):
    plt.figure(figsize=(8, 4.5))
    for name, d in results.items():
        if d["regrets"] is None:
            continue
        plt.plot(d["regrets"], label=name)
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative regret")
    plt.title("Cumulative regret over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots from bandit results.")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="../pictures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = load_results(args.results)

    plot_rewards(results, os.path.join(args.out_dir, "bandit_avg_rewards.png"))
    plot_regrets(results, os.path.join(args.out_dir, "bandit_cum_regret.png"))
    print(f"Saved plots to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

