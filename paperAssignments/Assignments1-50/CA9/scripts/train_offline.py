import argparse
import random

import numpy as np
import torch

from ..src.config import default_config
from ..src.algos.au_dmg import AUDMG


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--env", type=str, default="antmaze-medium-diverse-v2")
    parser.add_argument("--logdir", type=str, default="outputs/ca9")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    cfg = default_config()
    set_seed(cfg.seed)

    # Try to load D4RL dataset; fall back to synthetic data for smoke tests.
    try:
        from ..src.data.replay_buffer import ReplayBuffer

        rb = ReplayBuffer.from_d4rl(args.env)
        s_dim = rb.s.shape[1]
        a_dim = rb.a.shape[1]
    except Exception:
        # synthetic dataset
        from ..src.data.replay_buffer import ReplayBuffer

        print("D4RL dataset not available; creating synthetic dataset for smoke test.")
        s_dim = 10
        a_dim = 2
        N = 10000
        obs = np.random.randn(N, s_dim).astype(np.float32)
        acts = np.random.uniform(-1, 1, size=(N, a_dim)).astype(np.float32)
        rews = np.random.randn(N).astype(np.float32) * 0.1
        next_obs = obs + 0.1 * np.random.randn(N, s_dim).astype(np.float32)
        dones = np.zeros(N, dtype=np.float32)
        rb = ReplayBuffer(max_size=N)
        rb.load_from_arrays(obs, acts, rews, next_obs, dones)

    agent = AUDMG(s_dim, a_dim, cfg)

    # Logging
    import os
    import csv
    import time

    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, f"train_{int(time.time())}.csv")
    with open(log_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "step",
                "v_loss",
                "critic_loss",
                "policy_loss",
                "lam_mean",
                "std_mild_mean",
            ],
        )
        writer.writeheader()
        step = 0
        for epoch in range(args.epochs):
            iters = args.steps
            for it in range(iters):
                batch = rb.sample_batch(cfg.batch_size)
                stats = agent.update(batch)
                stats_row = {"step": step}
                stats_row.update(stats)
                writer.writerow(stats_row)
                if step % 100 == 0:
                    print(
                        f"step={step} v_loss={stats['v_loss']:.4f} critic={stats['critic_loss']:.4f} lam={stats['lam_mean']:.3f}"
                    )
                step += 1
    print(f"Training finished. Logs saved to {log_path}")


if __name__ == "__main__":
    main()
