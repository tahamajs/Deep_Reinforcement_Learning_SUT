"""
Visualization utilities for MaxSink (CA8).
Provides plotting functions for reward histograms, Sinkhorn loss curves, and particle PCA scatter.
"""

from typing import Sequence, Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

sns.set(style="whitegrid")


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_reward_histograms(
    raw_rewards: Sequence[float],
    transformed_rewards: Sequence[float],
    save_path: str = "pictures/reward_hist.png",
) -> None:
    """Plot raw vs transformed reward histograms side-by-side and save to file."""
    ensure_dir(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(raw_rewards, ax=axes[0], bins=30, kde=False, color="C0")
    axes[0].set_title("Raw Rewards")
    sns.histplot(transformed_rewards, ax=axes[1], bins=30, kde=False, color="C1")
    axes[1].set_title("Transformed Rewards")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_sinkhorn_loss(
    losses: Sequence[float], save_path: str = "pictures/sinkhorn_loss.png"
) -> None:
    """Plot Sinkhorn loss curve over training steps."""
    ensure_dir(save_path)
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Sinkhorn loss")
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.title("Sinkhorn Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_particle_pca(
    pred_particles: np.ndarray,
    target_particles: np.ndarray,
    save_path: str = "pictures/particles_pca.png",
    sample: Optional[int] = 500,
) -> None:
    """
    Project particles to 2D with PCA and plot predicted vs target particle clouds.
    pred_particles, target_particles: arrays of shape [B, N, d] or [N, d]
    """
    ensure_dir(save_path)

    # flatten if batched
    def flatten(p):
        p = np.asarray(p)
        if p.ndim == 3:
            p = p.reshape(-1, p.shape[-1])
        return p

    xp = flatten(pred_particles)
    yp = flatten(target_particles)
    # sample for speed
    if sample is not None and xp.shape[0] > sample:
        idx = np.random.choice(xp.shape[0], sample, replace=False)
        xp_s = xp[idx]
    else:
        xp_s = xp
    if sample is not None and yp.shape[0] > sample:
        idx = np.random.choice(yp.shape[0], sample, replace=False)
        yp_s = yp[idx]
    else:
        yp_s = yp

    all_xy = np.vstack([xp_s, yp_s])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_xy)
    xp_coords = coords[: xp_s.shape[0]]
    yp_coords = coords[xp_s.shape[0] :]

    plt.figure(figsize=(6, 6))
    plt.scatter(
        xp_coords[:, 0], xp_coords[:, 1], s=10, alpha=0.6, label="predicted", c="C0"
    )
    plt.scatter(
        yp_coords[:, 0], yp_coords[:, 1], s=10, alpha=0.6, label="target", c="C1"
    )
    plt.legend()
    plt.title("Particle PCA: predicted vs target")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Quick demo with synthetic data
    rng = np.random.RandomState(0)
    raw = rng.choice([0.0, 1.0], size=1000, p=[0.95, 0.05])
    transformed = np.clip(
        raw + rng.normal(scale=0.1, size=raw.shape) + (raw > 0) * 0.4, 0.0, 1.4
    )
    plot_reward_histograms(raw, transformed, save_path="pictures/demo_reward_hist.png")
    losses = np.abs(np.sin(np.linspace(0, 6.0, 500))) + np.linspace(0.2, 0.01, 500)
    plot_sinkhorn_loss(losses, save_path="pictures/demo_sinkhorn_loss.png")
    pred = rng.normal(size=(200, 32, 1))
    targ = pred + rng.normal(scale=0.3, size=pred.shape)
    plot_particle_pca(pred, targ, save_path="pictures/demo_particles_pca.png")






