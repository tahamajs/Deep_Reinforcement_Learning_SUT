"""Generate offline dataset with optional YAML config/scenario."""
from __future__ import annotations

import argparse

from projects.amasa_clean.scripts.common import add_common_config_flags, resolve_config
from projects.amasa_clean.scripts.pipelines import generate_dataset_pipeline


def main(args):
    cfg = resolve_config(args)
    max_steps = args.max_steps if args.max_steps is not None else cfg["env"]["max_steps"]
    count = generate_dataset_pipeline(cfg, out_path=args.out, episodes=args.episodes, max_steps=max_steps)
    print(f"Saved {count} transitions to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_config_flags(parser)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="data/amasa_offline.npz")
    args = parser.parse_args()
    main(args)
