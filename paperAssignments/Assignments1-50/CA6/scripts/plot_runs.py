"""
Plotting utilities for VAD-PPO logs.
Usage:
    python scripts/plot_runs.py logs/vadppo_log.csv out_gamma.png
"""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_gamma(log_csv: str, out: str):
    df = pd.read_csv(log_csv)
    plt.figure(figsize=(8, 3))
    plt.plot(df["update"], df["gamma"], label="gamma")
    if "varA" in df.columns:
        plt.plot(df["update"], df["varA"], label="varA")
    plt.xlabel("update")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)


def plot_returns(log_csv: str, out: str):
    df = pd.read_csv(log_csv)
    plt.figure(figsize=(8, 4))
    plt.plot(df["update"], df["return_mean"], label="return_mean")
    if "return_std" in df.columns:
        plt.fill_between(
            df["update"],
            df["return_mean"] - df["return_std"],
            df["return_mean"] + df["return_std"],
            alpha=0.2,
        )
    plt.xlabel("update")
    plt.ylabel("return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)


def plot_hist_advantages(adv_arr: np.ndarray, out: str, bins: int = 50):
    plt.figure(figsize=(6, 4))
    plt.hist(adv_arr.flatten(), bins=bins, density=True)
    plt.xlabel("advantage")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out, dpi=300)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("out")
    p.add_argument("--which", choices=["gamma", "returns"], default="gamma")
    args = p.parse_args()
    if args.which == "gamma":
        plot_gamma(args.csv, args.out)
    else:
        plot_returns(args.csv, args.out)


if __name__ == "__main__":
    main()


