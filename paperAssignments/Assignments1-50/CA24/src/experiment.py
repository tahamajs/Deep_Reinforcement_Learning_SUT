from __future__ import annotations

from dataclasses import asdict
import time
from typing import Dict

import torch

from .config import Config
from .model import SimpleMLP
from .data import get_dataloader
from .losses import WeightedMSE
from .utils import set_seed, get_device


def train_one_epoch(model: torch.nn.Module, dataloader, optimizer, loss_fn, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        count += 1
    return total_loss / max(1, count)


def run_experiment(config: Config) -> Dict[str, float]:
    """Run a short training loop returning final metrics.

    The function is import-safe and can be used by notebooks or scripts.
    """
    set_seed(config.seed)
    device = get_device(prefer_gpu=(config.device == "cuda"))

    dataloader = get_dataloader(batch_size=config.batch_size, input_dim=config.input_dim)
    model = SimpleMLP(config.input_dim, list(config.hidden_dims), config.output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = WeightedMSE()

    logs = {}
    for epoch in range(config.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device)
        logs[f"train/loss_epoch_{epoch}"] = train_loss
        logs[f"train/time_epoch_{epoch}"] = time.time() - t0

    # Return a compact summary
    summary = {
        "final_train_loss": logs[f"train/loss_epoch_{config.epochs-1}"]
    }
    return summary


if __name__ == "__main__":
    # Minimal CLI demo. Not executed during tests.
    cfg = Config()
    print("Running demo experiment with Config:", asdict(cfg))
    out = run_experiment(cfg)
    print("Finished. Summary:", out)
