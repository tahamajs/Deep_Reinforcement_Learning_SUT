"""
Launcher script: optionally load a YAML config then run training.
"""

import argparse
import os
import sys

# make src importable
THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML config to load")
    p.add_argument("--steps", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.config:
        from config_loader import load_config_from_yaml  # type: ignore

        load_config_from_yaml(args.config)
    # override steps if provided
    if args.steps is not None:
        from config import cfg  # type: ignore

        try:
            setattr(cfg, "total_steps", args.steps)  # type: ignore[attr-defined]
        except Exception:
            # recreate cfg dataclass
            from config import Config  # type: ignore

            new_vals = cfg.as_dict()
            new_vals["total_steps"] = args.steps
            globals()["cfg"] = Config(**new_vals)  # type: ignore

    # run training
    from scripts.train import main as train_main  # type: ignore

    train_main()


if __name__ == "__main__":
    main()
