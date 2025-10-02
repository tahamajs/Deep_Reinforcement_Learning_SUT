"""
Example script for running meta-learning training.

Usage:
    python scripts/train_meta.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from configs.meta_config import *
import run_hw5
def main():
    """Run meta-learning training with default configuration."""

    args = (
        ["meta", "--env_name", ENV_NAME, "--hidden_sizes"]
        + [str(size) for size in HIDDEN_SIZES]
        + [
            "--learning_rate",
            str(LEARNING_RATE),
            "--meta_learning_rate",
            str(META_LEARNING_RATE),
            "--adaptation_steps",
            str(ADAPTATION_STEPS),
            "--meta_batch_size",
            str(META_BATCH_SIZE),
            "--num_tasks",
            str(NUM_TASKS),
            "--meta_steps",
            str(META_STEPS),
            "--discount",
            str(DISCOUNT),
            "--max_episode_length",
            str(MAX_EPISODE_LENGTH),
            "--log_interval",
            str(LOG_INTERVAL),
        ]
    )
    sys.argv = ["run_hw5.py"] + args
    run_hw5.main()
if __name__ == "__main__":
    main()