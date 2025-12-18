"""Training script for CA15.

This module is import-safe and exposes a `train` function that runs a
simple training loop on the synthetic dataset. It is intentionally
minimal so it is safe to import in unit tests and notebooks.
"""
from __future__ import annotations

from typing import Optional
import argparse
import time

import torch
from torch import optim

from config import Config
from data import SyntheticDataset
from model import MLPPolicy, ValueNetwork
from losses import mse_loss, policy_gradient_loss
from utils import set_seed, save_checkpoint


def train(cfg: Config, save_path: Optional[str] = None) -> dict:
    """Run a minimal training loop and return a summary dict.

    This function is deterministic given the same `cfg.seed` and uses the
    synthetic dataset included in the assignment. It is intentionally
    lightweight so it can be run on CPU for quick checks.

    Returns a dictionary with final losses and timings.
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Prepare data and models
    ds = SyntheticDataset(cfg.input_dim, cfg.output_dim, size=1024, seed=cfg.seed)
    policy = MLPPolicy(cfg.input_dim, cfg.hidden_dim, cfg.output_dim).to(device)
    value = ValueNetwork(cfg.input_dim, cfg.hidden_dim).to(device)

    opt_policy = optim.Adam(policy.parameters(), lr=cfg.lr)
    opt_value = optim.Adam(value.parameters(), lr=cfg.lr)

    start_time = time.time()
    last_policy_loss = None
    last_value_loss = None

    for epoch in range(cfg.epochs):
        for s, a, v in ds.batches(cfg.batch_size):
            s = s.to(device)
            a = a.to(device)
            v = v.to(device)

            # Value update (MSE regression)
            preds = value(s)
            val_loss = mse_loss(preds, v)
            opt_value.zero_grad()
            val_loss.backward()
            opt_value.step()

            # Policy update (REINFORCE-like with advantage)
            logits = policy(s)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            logp = dist.log_prob(a)
            # simple advantage: (v - baseline) where baseline is current value preds
            with torch.no_grad():
                advantage = v - preds.detach()
            pol_loss = policy_gradient_loss(logp, advantage, reduction="mean")
            opt_policy.zero_grad()
            pol_loss.backward()
            opt_policy.step()

            last_policy_loss = pol_loss.item()
            last_value_loss = val_loss.item()

    elapsed = time.time() - start_time

    if save_path:
        save_checkpoint(save_path, policy, opt_policy, extra={"cfg": cfg.__dict__})

    summary = {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "time_sec": elapsed,
        "epochs": cfg.epochs,
    }
    return summary


def _parse_args():
    p = argparse.ArgumentParser(description="Run CA15 training (minimal)")
    p.add_argument("--cfg", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--save", default=None, help="Path to save checkpoint (optional)")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = Config.from_yaml(args.cfg)
    summary = train(cfg, save_path=args.save)
    print("Training finished:", summary)


if __name__ == "__main__":
    main()
