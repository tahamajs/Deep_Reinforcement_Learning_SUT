"""Small example experiment demonstrating training loop and saving outputs.

Run as: python -m scripts.run_experiment --config ../configs/default.yaml --out outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import torch.optim as optim

from src.config import load_config
from src.data import get_dataloader
from src.model import MLP
from src.losses import mse_loss
from src.utils import FitResult, set_seed, save_loss_curve, ensure_dir


def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = mse_loss(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += float(loss.item()) * x.shape[0]
        n += x.shape[0]
    return total_loss / max(1, n)


def fit(cfg, out_dir: Path) -> FitResult:
    device = torch.device(cfg.train.device)
    set_seed(cfg.train.seed)
    loader = get_dataloader(n_samples=1000, batch_size=cfg.train.batch_size, seed=cfg.train.seed)
    model = MLP(cfg.model.input_dim, cfg.model.hidden_dims, cfg.model.output_dim, activation=cfg.model.activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    losses = []
    for epoch in range(cfg.train.epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        losses.append(loss)
    # Save artifacts
    out_dir = ensure_dir(out_dir)
    torch.save(model.state_dict(), out_dir / "model.pt")
    save_loss_curve(losses, out_dir / "loss")
    return FitResult(losses=losses, final_state={"model_state_dict": model.state_dict()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    res = fit(cfg, Path(args.out))
    print(f"Done in {time.time()-t0:.1f}s; last loss={res.losses[-1]:.4f}")


if __name__ == "__main__":
    main()
