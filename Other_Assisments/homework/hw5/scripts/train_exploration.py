#!/usr/bin/env python3
"""
Example script for running exploration training.

Usage:
    python scripts/train_exploration.py
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from configs.exploration_config import *
import run_hw5


def main():
    """Run exploration training with default configuration."""
    # Create argument list
    args = [
        "exploration",
        "--env_name",
        ENV_NAME,
        "--bonus_coeff",
        str(BONUS_COEFF),
        "--initial_rollouts",
        str(INITIAL_ROLLOUTS),
        "--max_episode_length",
        str(MAX_EPISODE_LENGTH),
        "--log_interval",
        str(LOG_INTERVAL),
    ]

    # Run training
    sys.argv = ["run_hw5.py"] + args
    run_hw5.main()


if __name__ == "__main__":
    main()
