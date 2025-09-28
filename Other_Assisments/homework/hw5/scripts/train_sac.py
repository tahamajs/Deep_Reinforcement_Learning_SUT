#!/usr/bin/env python3
"""
Example script for running SAC training.

Usage:
    python scripts/train_sac.py
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from configs.sac_config import *
import run_hw5


def main():
    """Run SAC training with default configuration."""
    # Create argument list
    args = (
        ["sac", "--env_name", ENV_NAME, "--hidden_sizes"]
        + [str(size) for size in HIDDEN_SIZES]
        + [
            "--learning_rate",
            str(LEARNING_RATE),
            "--alpha",
            str(ALPHA),
            "--batch_size",
            str(BATCH_SIZE),
            "--discount",
            str(DISCOUNT),
            "--tau",
            str(TAU),
            "--total_steps",
            str(TOTAL_STEPS),
            "--max_episode_length",
            str(MAX_EPISODE_LENGTH),
            "--log_interval",
            str(LOG_INTERVAL),
        ]
    )

    if REPARAMETERIZE:
        args.append("--reparameterize")

    # Run training
    sys.argv = ["run_hw5.py"] + args
    run_hw5.main()


if __name__ == "__main__":
    main()
