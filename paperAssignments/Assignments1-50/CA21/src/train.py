from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import Config
from .data import SyntheticDataset
from .losses import policy_gradient_loss, value_mse_loss
from .model import MLPPolicy, MLPValue
from .utils import set_seed, save_checkpoint


def train(
    cfg: Optional[Config] = None,
    num_samples: int = 256,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """Run a short training loop and return simple metrics.

    This function is intentionally small and import-safe; it is suitable for
    use from notebooks and as a smoke-test in CI.

    Args:
        cfg: Optional Config instance. If None, a default Config is created.
        num_samples: number of synthetic samples to use for the demo run.
        checkpoint_path: if provided, saves a final checkpoint using `save_checkpoint`.

    Returns:
        A dictionary with training metrics such as final losses and timing info.
    """
    if cfg is None:
        cfg = Config()

    # reproducibility
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dataset = SyntheticDataset(num_samples=num_samples, input_dim=cfg.input_dim, action_dim=cfg.action_dim, seed=cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    policy = MLPPolicy(cfg.input_dim, cfg.hidden_dim, cfg.action_dim).to(device)
    value = MLPValue(cfg.input_dim, cfg.hidden_dim).to(device)

    opt_policy = optim.Adam(policy.parameters(), lr=cfg.lr)
    opt_value = optim.Adam(value.parameters(), lr=cfg.lr)

    t0 = time.time()
    last_pg_loss = 0.0
    last_v_loss = 0.0
    for epoch in range(cfg.epochs):
        for obs, actions, rewards, next_obs, dones in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)

            # forward
            logits = policy(obs)
            logp = policy.log_prob(obs, actions)
            values = value(obs)

            # make simple 'advantages' signal (reward - value) for demo
            advantages = (rewards - values).detach()

            # losses
            pg_loss = policy_gradient_loss(logp, advantages)
            v_loss = value_mse_loss(values, rewards)

            opt_policy.zero_grad()
            pg_loss.backward()
            opt_policy.step()

            opt_value.zero_grad()
            v_loss.backward()
            opt_value.step()

            last_pg_loss = float(pg_loss.detach().cpu().item())
            last_v_loss = float(v_loss.detach().cpu().item())

    dt = time.time() - t0

    metrics = {
        "final_pg_loss": last_pg_loss,
        "final_value_loss": last_v_loss,
        "seconds": dt,
    }

    if checkpoint_path is not None:
        state = {
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value.state_dict(),
            "config": asdict(cfg),
            "metrics": metrics,
        }
        save_checkpoint(checkpoint_path, state)

    return metrics


if __name__ == "__main__":
    # allow running as a quick smoke demo from the command line
    import argparse

    parser = argparse.ArgumentParser(prog="ca21_train")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = Config(seed=42, input_dim=8, hidden_dim=32, action_dim=4, lr=1e-3, batch_size=args.batch_size, epochs=args.epochs)
    print("Running training demo with config:", cfg)
    res = train(cfg=cfg, checkpoint_path=args.checkpoint)
    print("Done. Metrics:", res)
