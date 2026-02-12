"""Evaluate checkpoints on reward/safety and plot Pareto frontier."""
from __future__ import annotations

import argparse

from projects.amasa_clean.scripts.common import add_common_config_flags, resolve_config
from projects.amasa_clean.scripts.pipelines import evaluate_checkpoints_pipeline


def main(args):
    cfg = resolve_config(args)
    if args.episodes is not None:
        cfg["eval"]["episodes"] = args.episodes
    if args.max_steps is not None:
        cfg["env"]["max_steps"] = args.max_steps
    if args.device:
        cfg["experiment"]["device"] = args.device

    points = evaluate_checkpoints_pipeline(cfg, checkpoints_dir=args.checkpoints, out_plot=args.out)
    if not points:
        print("no valid checkpoints found")
        return
    for p in points:
        print(f"{p['checkpoint']}: reward {p['avg_reward']:.1f}, cost {p['avg_cost']:.3f}")
    print("Saved plot to", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_config_flags(parser)
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="plots/pareto.png")
    args = parser.parse_args()
    main(args)
