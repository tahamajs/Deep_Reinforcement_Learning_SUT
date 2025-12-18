"""Visualization utilities for SimGolf branches and checkpoints.

Usage examples (script):
    python analysis/visualize_branches.py branches.json ../pictures/fig_branch_returns.png

This script expects a JSON file containing a list of branches where each branch
is a dict with keys: 'ret' and 'traj' (traj is list of [z, a, r, gamma]; z/a may be lists).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_branches(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def plot_branch_returns(branches: List[Any], out: str):
    rets = [float(b["ret"]) for b in branches]
    plt.figure(figsize=(6, 4))
    plt.hist(rets, bins=20, color="#4C72B0", alpha=0.9)
    plt.xlabel("Branch return")
    plt.ylabel("Count")
    plt.title("Distribution of branch returns")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


def plot_topk_actions(branches: List[Any], out: str, topk: int = 8):
    # collect first actions of top-k branches
    branches_sorted = sorted(branches, key=lambda b: float(b["ret"]), reverse=True)
    selected = branches_sorted[:topk]
    acts = []
    for b in selected:
        if len(b["traj"]) == 0:
            continue
        a0 = b["traj"][0][1]
        acts.append(np.array(a0).ravel())
    if len(acts) == 0:
        return
    acts = np.stack(acts)
    # if action dim >2, scatter first two dims
    plt.figure(figsize=(5, 5))
    if acts.shape[1] >= 2:
        plt.scatter(acts[:, 0], acts[:, 1], c="C1")
        plt.xlabel("action dim 0")
        plt.ylabel("action dim 1")
    else:
        plt.bar(np.arange(len(acts)), acts[:, 0])
        plt.ylabel("action value")
    plt.title(f"Top-{topk} branch first actions")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


def plot_branch_latent_pca(branches: List[Any], out: str, max_samples: int = 200):
    # flatten all z in trajectories
    zs = []
    for b in branches:
        for step in b["traj"]:
            z = step[0]
            zs.append(np.array(z).ravel())
            if len(zs) >= max_samples:
                break
        if len(zs) >= max_samples:
            break
    if len(zs) < 2:
        return
    zs = np.stack(zs)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(zs)
    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.7)
    plt.title("PCA of imagined latents (sampled)")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: visualize_branches.py branches.json out_prefix")
        sys.exit(1)
    branches = load_branches(sys.argv[1])
    out_prefix = sys.argv[2]
    plot_branch_returns(branches, out_prefix.replace(".png", "_returns.png"))
    plot_topk_actions(branches, out_prefix.replace(".png", "_actions.png"))
    plot_branch_latent_pca(branches, out_prefix.replace(".png", "_pca.png"))









