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
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--tb-logdir", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="au_dmg")
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
    ckpt_dir = os.path.join(args.logdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # optional TensorBoard
    tb_writer = None
    if args.tb_logdir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=args.tb_logdir)
        except Exception as e:
            print("TensorBoard SummaryWriter unavailable:", e)
            tb_writer = None
    # optional wandb
    wandb = None
    if args.use_wandb:
        try:
            import wandb

            wandb.init(project=args.wandb_project, config=cfg.__dict__)
        except Exception as e:
            print("wandb init failed:", e)
            wandb = None
    if args.resume:
        try:
            agent.load_checkpoint(args.resume)
            print(f"Resumed checkpoint from {args.resume}")
        except Exception as e:
            print("Failed to load checkpoint:", e)
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
        # histories for plotting
        hist_steps = []
        hist_v = []
        hist_critic = []
        hist_policy = []
        hist_lam = []
        hist_std = []
        for epoch in range(args.epochs):
            iters = args.steps
            for it in range(iters):
                batch = rb.sample_batch(cfg.batch_size)
                stats = agent.update(batch)
                stats_row = {"step": step}
                stats_row.update(stats)
                writer.writerow(stats_row)
                # record histories
                hist_steps.append(step)
                hist_v.append(stats["v_loss"])
                hist_critic.append(stats["critic_loss"])
                hist_policy.append(stats["policy_loss"])
                hist_lam.append(stats["lam_mean"])
                hist_std.append(stats["std_mild_mean"])
                # tensorboard
                if tb_writer is not None:
                    tb_writer.add_scalar("loss/v_loss", stats["v_loss"], step)
                    tb_writer.add_scalar("loss/critic_loss", stats["critic_loss"], step)
                    tb_writer.add_scalar("misc/lam_mean", stats["lam_mean"], step)
                    tb_writer.add_scalar(
                        "misc/std_mild_mean", stats["std_mild_mean"], step
                    )
                # wandb
                if args.use_wandb:
                    try:
                        import wandb as _wandb

                        _wandb.log({"step": step, **stats})
                    except Exception:
                        pass
                if step % 100 == 0:
                    print(
                        f"step={step} v_loss={stats['v_loss']:.4f} critic={stats['critic_loss']:.4f} lam={stats['lam_mean']:.3f}"
                    )
                step += 1
                if args.save_interval > 0 and (step % args.save_interval == 0):
                    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{step}.pth")
                    try:
                        agent.save_checkpoint(ckpt_path)
                        print(f"Saved checkpoint: {ckpt_path}")
                    except Exception as e:
                        print("Failed to save checkpoint:", e)
    print(f"Training finished. Logs saved to {log_path}")
    # save history plots and config snapshot
    try:
        from ..src.utils.logger import plot_series
        out_dir = os.path.join(args.logdir, "plots")
        os.makedirs(out_dir, exist_ok=True)
        if len(hist_steps) > 0:
            plot_series(hist_steps, {"v_loss": hist_v, "critic_loss": hist_critic, "policy_loss": hist_policy}, os.path.join(out_dir, "losses.png"), title="Losses")
            plot_series(hist_steps, {"lam_mean": hist_lam, "std_mild_mean": hist_std}, os.path.join(out_dir, "lam_std.png"), title="Lambda and Std")
    except Exception:
        pass
    try:
        import json
        cfg_path = os.path.join(args.logdir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg.__dict__, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
