"""
Plotting utilities for CA12 (RA-U-OBAC).

Produces publication-quality figures (matplotlib / seaborn) from demo outputs:
- eval_returns.csv -> learning curve
- optional logs/ loss CSVs if available

Usage:
    python analysis/plot_results.py --ckpt_dir outputs/ca12_checkpoints --out_dir pictures
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", context="paper", font_scale=1.1)


def plot_eval_returns(csv_path: Path, out_dir: Path) -> Path:
    """Plot eval returns over steps from eval_returns.csv and save figure."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"No eval data at {csv_path}")

    # smooth with rolling median for visual clarity
    df["smoothed"] = (
        df["avg_return"]
        .rolling(window=max(1, int(len(df) / 10)), min_periods=1)
        .median()
    )

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(df["steps"], df["avg_return"], color="#1f77b4", alpha=0.25, label="raw")
    plt.plot(
        df["steps"], df["smoothed"], color="#1f77b4", lw=2.0, label="smoothed (median)"
    )
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.title("RA-U-OBAC: Eval Return vs Steps")
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_eval_returns.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_loss_csv(loss_csv: Path, out_dir: Path, col_prefix: str = "loss"):
    """Generic loss CSV plotting (expects columns like loss_total, loss_pi, loss_v, loss_prefix)."""
    df = pd.read_csv(loss_csv)
    if df.empty:
        raise RuntimeError(f"No loss data at {loss_csv}")

    # select columns starting with prefix
    cols = [c for c in df.columns if c.startswith(col_prefix)]
    if not cols:
        raise RuntimeError(f"No columns starting with {col_prefix} in {loss_csv}")

    plt.figure(figsize=(7, 4))
    for c in cols:
        plt.plot(
            df["step"] if "step" in df.columns else np.arange(len(df)), df[c], label=c
        )
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Losses")
    plt.legend(ncol=2, fontsize="small")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_losses.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=Path, default=Path("outputs/ca12_checkpoints"))
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("paperAssignments/Assignments1-50/CA12/pictures"),
    )
    p.add_argument(
        "--loss_csv", type=Path, default=None, help="Optional training loss CSV"
    )
    args = p.parse_args(argv)

    eval_csv = args.ckpt_dir / "eval_returns.csv"
    if eval_csv.exists():
        out = plot_eval_returns(eval_csv, args.out_dir)
        print(f"Saved eval plot to {out}")
    else:
        print(f"No eval_returns.csv found in {args.ckpt_dir}")

    if args.loss_csv:
        if args.loss_csv.exists():
            out = plot_loss_csv(args.loss_csv, args.out_dir)
            print(f"Saved loss plot to {out}")
        else:
            print(f"Loss CSV {args.loss_csv} not found")


if __name__ == "__main__":
    main()









