"""Train offline algorithm (CQL/IQL) with backward-compatible CLI."""
from __future__ import annotations

import argparse

from projects.amasa_clean.scripts.common import add_common_config_flags, resolve_config
from projects.amasa_clean.scripts.pipelines import train_offline_pipeline


def main(args):
    cfg = resolve_config(args)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.log_every is not None:
        cfg["train"]["eval_every"] = args.log_every
    if args.save_every is not None:
        cfg["train"]["save_every"] = args.save_every
    if args.device:
        cfg["experiment"]["device"] = args.device
    if args.cql_alpha is not None:
        cfg["algo"]["cql_alpha"] = args.cql_alpha

    metrics = train_offline_pipeline(cfg, dataset_path=args.dataset, out_dir=args.out_dir)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_config_flags(parser)
    parser.add_argument("--dataset", type=str, default="data/amasa_offline.npz")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cql_alpha", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    args = parser.parse_args()
    main(args)
