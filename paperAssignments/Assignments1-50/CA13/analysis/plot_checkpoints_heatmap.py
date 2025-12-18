"""Plot heatmap / embedding of checkpoint latents."""

from __future__ import annotations
from pathlib import Path
from typing import List, Any
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_checkpoints(path: str) -> List[Any]:
    with open(path, "r") as f:
        return json.load(f)


def plot_checkpoint_tsne(checkpoints: List[Any], out: str, perplexity: int = 30):
    zs = [np.array(c["z"]).ravel() for c in checkpoints if "z" in c]
    if len(zs) < 2:
        return
    zs = np.stack(zs)
    emb = TSNE(
        n_components=2, perplexity=perplexity, init="pca", random_state=0
    ).fit_transform(zs)
    plt.figure(figsize=(6, 5))
    plt.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.8)
    plt.title("t-SNE of checkpoint latents")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: plot_checkpoints_heatmap.py checkpoints.json out.png")
        raise SystemExit(1)
    ckpts = load_checkpoints(sys.argv[1])
    plot_checkpoint_tsne(ckpts, sys.argv[2])
