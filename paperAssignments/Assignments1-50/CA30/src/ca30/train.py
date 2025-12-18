from typing import Tuple, Dict, Any
import numpy as np
from .model import BaseModel
from .utils import make_rng, save_json, ensure_dir
from pathlib import Path
import matplotlib.pyplot as plt


def make_synthetic_batch(batch_size: int, input_dim: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.randn(batch_size, input_dim).astype(np.float32)
    y = (np.sum(X, axis=1) > 0).astype(int)
    return X, y


def simple_train_epoch(model: BaseModel, batch_size: int, input_dim: int, rng: np.random.RandomState):
    X, y = make_synthetic_batch(batch_size, input_dim, rng)
    out = model.forward(X)
    # tiny dummy update for numpy model
    if model.backend == "numpy":
        targets = np.zeros_like(out)
        targets[np.arange(len(y)), y] = 1.0
        grad = 2 * (out - targets) / out.shape[0]
        h = np.maximum(X.dot(model.W1) + model.b1, 0.0)
        dW2 = h.T.dot(grad)
        model.W2 -= 1e-3 * dW2
    return out


def train(cfg, model: BaseModel, rng: np.random.RandomState, max_steps: int = 5, out_dir: str | Path = "results/demo") -> Dict[str, Any]:
    """Run a tiny deterministic training run (fast and safe for CI).

    Returns a dict with metrics and writes a small learning curve PNG to `out_dir`.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    history = {"loss": []}
    for step in range(max_steps):
        out = simple_train_epoch(model, batch_size=cfg.batch_size, input_dim=cfg.input_dim, rng=rng)
        loss = float((out ** 2).mean())
        history["loss"].append(loss)

    # save metrics
    save_json(out_dir / "metrics.json", history)

    # save a small figure
    plt.figure(figsize=(4, 3))
    plt.plot(history["loss"], marker="o")
    plt.title("Demo learning curve")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    fig_path = out_dir / "figure1.png"
    plt.savefig(fig_path, dpi=100)
    plt.close()

    return {"history": history, "out_dir": str(out_dir)}