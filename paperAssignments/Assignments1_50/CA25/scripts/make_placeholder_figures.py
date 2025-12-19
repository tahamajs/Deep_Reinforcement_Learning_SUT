"""Generate illustrative placeholder figures for CA25 outputs/pictures.

Run:
    python scripts/make_placeholder_figures.py --out outputs/example_run/pictures

This script only uses matplotlib and numpy and creates example loss and prediction plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def make_loss_plot(path: Path):
    epochs = np.arange(1, 11)
    train = np.exp(-0.3 * epochs) + 0.02 * np.random.rand(len(epochs))
    val = np.exp(-0.25 * epochs) + 0.03 * np.random.rand(len(epochs))
    plt.figure()
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Placeholder: training & validation loss")
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / "loss.png", dpi=150)
    plt.close()


def make_predictions_plot(path: Path):
    rng = np.random.RandomState(0)
    true = rng.randn(200)
    pred = true + 0.2 * rng.randn(200)
    plt.figure()
    plt.scatter(true, pred, alpha=0.6, s=8)
    mn = min(true.min(), pred.min())
    mx = max(true.max(), pred.max())
    plt.plot([mn, mx], [mn, mx], c="k", lw=1, ls="--")
    plt.xlabel("true")
    plt.ylabel("predicted")
    plt.title("Placeholder: true vs predicted")
    plt.savefig(path / "predictions.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/example_run/pictures")
    args = p.parse_args()
    out = Path(args.out)
    make_loss_plot(out)
    make_predictions_plot(out)
    print(f"Wrote placeholder figures to {out}")
