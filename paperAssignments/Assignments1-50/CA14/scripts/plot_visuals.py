"""Plotting utilities to generate publication-quality figures for the MAMBA-PEAC paper.

This script contains functions that, given real training logs and latents, will
produce the following figures and save them to ../pictures/:
- fig_01_tsne_z_morph.png : t-SNE/UMAP of inferred morphology latents colored by morphology id
- fig_02_adaptation_curve.png : few-shot adaptation curves (returns vs episode)
- fig_03_reconstruction.png : reconstruction examples (obs vs recon)
- fig_04_kl_trends.png : KL(z) and KL(z_morph) over training steps

The functions are ready-to-run; they do not execute upon import. Fill the data-loading
parts to point to your actual logs/checkpoints.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

PICS_DIR = Path(__file__).resolve().parents[1] / "pictures"
PICS_DIR.mkdir(parents=True, exist_ok=True)


def plot_tsne_z_morph(
    z_morph: np.ndarray,
    morph_ids: np.ndarray,
    filename: str = "fig_01_tsne_z_morph.png",
) -> None:
    """Plot 2D t-SNE of morphology latents.

    Args:
        z_morph: (N, D) array of inferred morphology vectors
        morph_ids: (N,) array of integer/string labels for morphology
    """
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    z2 = tsne.fit_transform(z_morph)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=z2[:, 0], y=z2[:, 1], hue=morph_ids, palette="tab10", s=20, linewidth=0
    )
    plt.title("t-SNE of inferred morphology latents")
    plt.axis("off")
    out = PICS_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def plot_adaptation_curve(
    returns: np.ndarray, labels: list, filename: str = "fig_02_adaptation_curve.png"
) -> None:
    """Plot adaptation curves.

    Args:
        returns: array shaped (n_methods, episodes) or list of arrays
        labels: list of labels per method
    """
    plt.figure(figsize=(6, 4))
    if isinstance(returns, list):
        for r, lab in zip(returns, labels):
            mean = np.mean(r, axis=0) if r.ndim == 2 else r
            plt.plot(mean, label=lab)
    else:
        for i, lab in enumerate(labels):
            plt.plot(np.mean(returns[i], axis=0), label=lab)
    plt.xlabel("Adaptation episode")
    plt.ylabel("Return")
    plt.legend()
    plt.title("Few-shot adaptation curves")
    out = PICS_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def plot_reconstruction(
    obs: np.ndarray,
    recon: np.ndarray,
    n: int = 5,
    filename: str = "fig_03_reconstruction.png",
) -> None:
    """Plot side-by-side reconstructions for proprio/state vectors (line plots).

    Args:
        obs: (N, D) or (T, D)
        recon: same shape as obs
    """
    plt.figure(figsize=(8, 2 * n))
    for i in range(n):
        ax = plt.subplot(n, 1, i + 1)
        ax.plot(obs[i], label="obs")
        ax.plot(recon[i], label="recon", linestyle="--")
        ax.set_ylabel(f"dim")
        if i == 0:
            ax.legend()
    plt.suptitle("Reconstruction examples (state dims)")
    out = PICS_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def plot_kl_trends(
    kl_z: np.ndarray, kl_m: np.ndarray, filename: str = "fig_04_kl_trends.png"
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(kl_z, label="KL z")
    plt.plot(kl_m, label="KL z_morph")
    plt.xlabel("Training step")
    plt.ylabel("KL (nats)")
    plt.legend()
    plt.title("KL trends during training")
    out = PICS_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print(
        "plot_visuals.py: library of plotting functions. Import and call functions with real data."
    )






