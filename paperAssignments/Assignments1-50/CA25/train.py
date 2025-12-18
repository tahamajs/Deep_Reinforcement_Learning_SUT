"""Lightweight training script for CA25 example.

Run from the repo root::
    python -m paperAssignments.Assignments1-50.CA25.train --config configs/example.yaml

The script is intentionally small and import-safe: nothing runs at import time.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from src.config import load_config, save_config
from src.utils import set_seed, get_device, ensure_dir, setup_logger
from src.data import get_dataloader
from src.model import MLP
from src.losses import regression_loss, classification_loss


logger = setup_logger(__name__)


def train_one_epoch(model, loader, opt, device, loss_fn):
    model.train()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.detach().cpu().item()) * X.shape[0]
        n += X.shape[0]
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        loss = loss_fn(out, y)
        total_loss += float(loss.detach().cpu().item()) * X.shape[0]
        n += X.shape[0]
    return total_loss / max(1, n)


def run_training(cfg_path: str | Path):
    cfg = load_config(cfg_path)
    save_dir = ensure_dir(cfg.save_dir)
    save_config(cfg, Path(save_dir) / "used_config.yaml")

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")

    train_loader, val_loader = get_dataloader(task=cfg.task, batch_size=cfg.batch_size, input_dim=cfg.input_dim, seed=cfg.seed)

    if cfg.task == "classification":
        output_activation = None
        output_dim = cfg.output_dim
        loss_fn = classification_loss
    else:
        output_activation = None
        output_dim = cfg.output_dim
        loss_fn = regression_loss

    model = MLP(cfg.input_dim, cfg.hidden_dims, output_dim, output_activation).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"train_loss": [], "val_loss": []}
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t_loss = train_one_epoch(model, train_loader, opt, device, loss_fn)
        v_loss = evaluate(model, val_loader, device, loss_fn)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        logger.info(f"Epoch {epoch:03d}/{cfg.epochs} — train {t_loss:.4f} — val {v_loss:.4f}")

    logger.info(f"Training finished in {time.time() - start:.1f}s")

    # save model
    torch.save(model.state_dict(), Path(save_dir) / "model.pt")

    # plot losses
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    pictures = ensure_dir(Path(save_dir) / "pictures")
    figpath = pictures / "loss.png"
    plt.savefig(figpath)
    logger.info(f"Saved loss plot to {figpath}")

    return model, history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/example.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
