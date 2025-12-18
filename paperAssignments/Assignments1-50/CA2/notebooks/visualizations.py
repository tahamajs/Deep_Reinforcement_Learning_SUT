"""
Notebook-style plotting utilities for CA2 (CrossQ + Sophia).
This script contains plotting functions to generate paper-quality figures
from CSV logs. Intended to be imported or run inside a Jupyter notebook.
Do NOT run automatically; inspect and run cells manually.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="paper", font_scale=1.1)

OUT_DIR = Path(__file__).resolve().parents[2] / "pictures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- I/O helpers ------------------------------------------------------------


def load_log(csv_path):
    """Load a CSV log produced by training.
    Expected columns (at least): step, env_step, seed, return, loss, grad_norm, hessian_mean, bn_mean_X, bn_var_X, step_time, utd
    The loader is permissive: missing columns are ignored.
    """
    return pd.read_csv(csv_path)


# -- Plotting functions -----------------------------------------------------


def plot_learning_curves(
    logs: list, label_col="label", x="env_step", y="return", smooth=10, out_file=None
):
    """Plot learning curves for multiple runs.

    logs: list of pandas.DataFrame with at least columns [x,y,seed]
    label_col: column name in each frame used to group (e.g., optimizer or utd)
    """
    plt.figure(figsize=(6.5, 3.5))
    for df in logs:
        if label_col in df:
            group = df[label_col].iloc[0]
        else:
            group = "run"
        # compute rolling mean across env steps grouped by seed
        if "seed" in df.columns:
            for seed, g in df.groupby("seed"):
                g = g.sort_values(x)
                plt.plot(
                    g[x],
                    g[y].rolling(smooth, min_periods=1).mean(),
                    alpha=0.25,
                    label=f"{group}-s{seed}",
                )
        else:
            df_sorted = df.sort_values(x)
            plt.plot(
                df_sorted[x],
                df_sorted[y].rolling(smooth, min_periods=1).mean(),
                alpha=0.8,
                label=str(group),
            )

    plt.xlabel("Env steps")
    plt.ylabel("Episode return")
    plt.legend(fontsize=8)
    plt.tight_layout()
    if out_file is None:
        out_file = OUT_DIR / "learning_curves.png"
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_norms(log_df: pd.DataFrame, out_file=None, smooth=50):
    """Plot gradient and Hessian norm trajectories (log-scale recommended)."""
    plt.figure(figsize=(6.5, 3.0))
    if "grad_norm" in log_df.columns:
        g = log_df.sort_values("step")
        plt.plot(
            g["step"],
            g["grad_norm"].rolling(smooth, min_periods=1).mean(),
            label="grad norm",
        )
    if "hessian_mean" in log_df.columns:
        h = log_df.sort_values("step")
        plt.plot(
            h["step"],
            h["hessian_mean"].rolling(smooth, min_periods=1).mean(),
            label="hessian mean",
        )
    plt.yscale("log")
    plt.xlabel("Optimizer step")
    plt.ylabel("Norm (log)")
    plt.legend()
    plt.tight_layout()
    if out_file is None:
        out_file = OUT_DIR / "norms.png"
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_bn_drift(log_df: pd.DataFrame, layer_prefix="bn", out_file=None):
    """Plot BN running mean/var drift across the training.
    Expects columns like 'bn1_running_mean', 'bn1_running_var', etc.
    """
    bn_mean_cols = [c for c in log_df.columns if "bn" in c and "mean" in c]
    bn_var_cols = [c for c in log_df.columns if "bn" in c and "var" in c]

    plt.figure(figsize=(6.5, 3.5))
    for col in bn_mean_cols[:6]:
        plt.plot(log_df["step"], log_df[col], alpha=0.8, label=col)
    plt.xlabel("Optimizer step")
    plt.ylabel("BN running mean")
    plt.legend(fontsize=7)
    plt.tight_layout()
    if out_file is None:
        out_file = OUT_DIR / "bn_running_mean.png"
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_update_magnitude_hist(updates_df: pd.DataFrame, out_file=None):
    """Plot histogram of update magnitudes (|delta theta| aggregated across params)."""
    if "update_mag" not in updates_df.columns:
        # try to compute from accumulated per-step info if present
        raise ValueError("updates_df must contain an update_mag column")
    plt.figure(figsize=(4.5, 3.0))
    sns.histplot(np.log1p(updates_df["update_mag"]), bins=50)
    plt.xlabel("log(1 + update magnitude)")
    plt.tight_layout()
    if out_file is None:
        out_file = OUT_DIR / "update_magnitude_hist.png"
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_step_time_vs_utd(log_df: pd.DataFrame, out_file=None):
    """Scatter plot step time vs UTD (assumes columns 'utd' and 'step_time')."""
    plt.figure(figsize=(5.5, 3.0))
    sns.boxplot(x="utd", y="step_time", data=log_df)
    plt.xlabel("UTD")
    plt.ylabel("Step time (s)")
    plt.tight_layout()
    if out_file is None:
        out_file = OUT_DIR / "step_time_vs_utd.png"
    plt.savefig(out_file, dpi=300)
    plt.close()


# -- Example usage (text only) ----------------------------------------------
#
# from pathlib import Path
# logs = [load_log('runs/sophia_utd10_seed0/log.csv'), load_log('runs/adam_utd10_seed0/log.csv')]
# for i, df in enumerate(logs):
#     df['label'] = ['Sophia' if i == 0 else 'Adam'][0]
# plot_learning_curves(logs, label_col='label', out_file=OUT_DIR/'learning_curves.png')
# plot_norms(logs[0], out_file=OUT_DIR/'norms_sophia.png')
# plot_bn_drift(logs[0], out_file=OUT_DIR/'bn_mean_sophia.png')
# plot_update_magnitude_hist(updates_df, out_file=OUT_DIR/'update_hist.png')
# plot_step_time_vs_utd(pd.concat(logs), out_file=OUT_DIR/'step_time_utd.png')

# End of visualizations module


