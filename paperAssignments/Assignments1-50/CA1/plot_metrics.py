"""Plotting utilities for CA1 experiments.

Expect a run directory with metrics CSV (columns: step, loss, w1, epsilon) and
optionally a particles npz file with arrays 'particles' shaped (steps, B, A, N, D)
or single-run particles (B, A, N, D). The script saves publication-ready PNGs.

Usage:
    python plot_metrics.py --log runs/ca1/logs.csv --particles runs/ca1/particles.npz --outdir runs/ca1/plots
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def plot_loss(log_df: pd.DataFrame, outpath: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(log_df["step"], log_df["loss"], label="Sinkhorn loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss (Sinkhorn)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_w1(log_df: pd.DataFrame, outpath: Path):
    if "w1" not in log_df.columns:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(log_df["step"], log_df["w1"], label="W1 (pred vs MC)")
    plt.xlabel("Step")
    plt.ylabel("W1")
    plt.title("Wasserstein-1 Distance vs MC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_epsilon(log_df: pd.DataFrame, outpath: Path):
    if "epsilon" not in log_df.columns:
        return
    plt.figure(figsize=(6, 3))
    plt.plot(log_df["step"], log_df["epsilon"], label="epsilon")
    plt.xlabel("Step")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Annealing Schedule")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_particles_scatter(particles: np.ndarray, outpath: Path, action: int = 0):
    # particles shape: (N, D) or (B, A, N, D) or (steps, B, A, N, D)
    # reduce to (N, D) by taking last step and first batch if needed
    if particles.ndim == 5:
        arr = particles[-1, 0, action]
    elif particles.ndim == 4:
        # (B, A, N, D)
        arr = particles[0, action]
    elif particles.ndim == 2:
        arr = particles
    else:
        raise ValueError("Unexpected particles shape: " + str(particles.shape))

    N, D = arr.shape
    plt.figure(figsize=(5, 5))
    if D == 1:
        sns.kdeplot(arr.squeeze(), fill=True)
        plt.xlabel("Return")
        plt.title(f"Particle KDE (action={action})")
    elif D == 2:
        plt.scatter(arr[:, 0], arr[:, 1], s=10, alpha=0.7)
        plt.xlabel("dim0")
        plt.ylabel("dim1")
        plt.title(f"Particle scatter (action={action})")
    else:
        # project by PCA-like mean and first principal component
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        proj = pca.fit_transform(arr)
        plt.scatter(proj[:, 0], proj[:, 1], s=10, alpha=0.7)
        plt.title(f"Particle PCA projection (action={action})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--particles", type=str, default=None)
    p.add_argument("--outdir", type=str, default="plots")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.log is not None:
        df = pd.read_csv(args.log)
        if "loss" in df.columns:
            plot_loss(df, outdir / "loss.png")
        plot_w1(df, outdir / "w1.png")
        plot_epsilon(df, outdir / "epsilon.png")

    if args.particles is not None:
        data = np.load(args.particles)
        # try common keys
        key = None
        for k in ["particles", "arr", "z"]:
            if k in data:
                key = k
                break
        if key is None:
            # pick first array
            key = list(data.files)[0]
        particles = data[key]
        # plot for action 0
        plot_particles_scatter(particles, outdir / "particles_action0.png", action=0)

    print("Plots saved to", outdir)


if __name__ == "__main__":
    main()










