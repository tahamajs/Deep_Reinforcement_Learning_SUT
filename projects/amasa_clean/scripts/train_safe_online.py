"""Train safe online algorithms with hybrid safety guard and backward-compatible CLI."""
from __future__ import annotations

import argparse

from projects.amasa_clean.scripts.common import add_common_config_flags, resolve_config
from projects.amasa_clean.scripts.pipelines import train_online_pipeline


def main(args):
    cfg = resolve_config(args)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.buffer_size is not None:
        cfg["train"]["buffer_size"] = args.buffer_size
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.random_steps is not None:
        cfg["train"]["random_steps"] = args.random_steps
    if args.shield_train_after is not None:
        cfg["train"]["shield_train_after"] = args.shield_train_after
    if args.log_every is not None:
        cfg["train"]["eval_every"] = args.log_every
    if args.save_every is not None:
        cfg["train"]["save_every"] = args.save_every
    if args.max_steps is not None:
        cfg["env"]["max_steps"] = args.max_steps

    cfg["safety"]["cost_limit"] = args.cost_limit if args.cost_limit is not None else cfg["safety"]["cost_limit"]
    cfg["safety"]["kp"] = args.kp if args.kp is not None else cfg["safety"]["kp"]
    cfg["safety"]["ki"] = args.ki if args.ki is not None else cfg["safety"]["ki"]
    cfg["safety"]["kd"] = args.kd if args.kd is not None else cfg["safety"]["kd"]
    cfg["safety"]["lambda_max"] = args.lambda_max if args.lambda_max is not None else cfg["safety"]["lambda_max"]
    cfg["safety"]["shield"]["enabled"] = bool(args.use_shield)

    if args.device:
        cfg["experiment"]["device"] = args.device
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    if not args.algo:
        # Backward-compatible default for this script is safe SAC online.
        cfg["algo"]["name"] = "sac_lag"

    metrics = train_online_pipeline(cfg, out_dir=args.out_dir, checkpoint_path=args.checkpoint)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_config_flags(parser)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--buffer_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--random_steps", type=int, default=None)
    parser.add_argument("--cost_limit", type=float, default=0.0)
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--ki", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--lambda_max", type=float, default=None)
    parser.add_argument("--cql_alpha", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--use_shield", action="store_true")
    parser.add_argument("--shield_train_after", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    args = parser.parse_args()
    main(args)
