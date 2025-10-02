"""
Logging Utilities

This module provides logging functionality for RL experiments.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import os
import json
import time
from collections import defaultdict
import numpy as np
class Logger:
    """Simple logger for RL experiments."""

    def __init__(self, output_dir=None, exp_name=None):
        """Initialize logger.

        Args:
            output_dir: Output directory for logs
            exp_name: Experiment name
        """
        self.output_dir = output_dir or "logs"
        self.exp_name = exp_name or f"exp_{int(time.time())}"
        self.log_dir = os.path.join(self.output_dir, self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.scalar_file = os.path.join(self.log_dir, "scalars.jsonl")
        self.config_file = os.path.join(self.log_dir, "config.json")
        self.scalars = defaultdict(list)
        self.config = {}
        self.step = 0

    def log_config(self, config_dict):
        """Log configuration parameters.

        Args:
            config_dict: Dictionary of configuration parameters
        """
        self.config.update(config_dict)
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)

    def log_scalar(self, key, value, step=None):
        """Log a scalar value.

        Args:
            key: Scalar name
            value: Scalar value
            step: Step number (uses internal counter if None)
        """
        if step is None:
            step = self.step
            self.step += 1
        self.scalars[key].append((step, value))
        log_entry = {"step": step, "key": key, "value": value, "timestamp": time.time()}

        with open(self.scalar_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")

    def log_scalars(self, scalar_dict, step=None):
        """Log multiple scalars.

        Args:
            scalar_dict: Dictionary of scalar names to values
            step: Step number
        """
        for key, value in scalar_dict.items():
            self.log_scalar(key, value, step)

    def get_scalar_history(self, key):
        """Get history of a scalar.

        Args:
            key: Scalar name

        Returns:
            List of (step, value) tuples
        """
        return self.scalars.get(key, [])

    def get_scalar_stats(self, key, window=100):
        """Get statistics for a scalar over recent window.

        Args:
            key: Scalar name
            window: Window size for statistics

        Returns:
            Dictionary with mean, std, min, max
        """
        history = self.get_scalar_history(key)
        if len(history) == 0:
            return {}

        recent_values = [v for _, v in history[-window:]]
        return {
            "mean": np.mean(recent_values),
            "std": np.std(recent_values),
            "min": np.min(recent_values),
            "max": np.max(recent_values),
            "count": len(recent_values),
        }

    def save_checkpoint(self, data, filename):
        """Save checkpoint data.

        Args:
            data: Data to save
            filename: Checkpoint filename
        """
        path = os.path.join(self.log_dir, filename)
        np.savez(path, **data)

    def load_checkpoint(self, filename):
        """Load checkpoint data.

        Args:
            filename: Checkpoint filename

        Returns:
            Loaded data dictionary
        """
        path = os.path.join(self.log_dir, filename)
        return np.load(path)

    def print_stats(self, step_interval=100):
        """Print current statistics.

        Args:
            step_interval: How often to print stats
        """
        if self.step % step_interval == 0:
            print(f"Step {self.step}:")
            for key in self.scalars.keys():
                stats = self.get_scalar_stats(key)
                if stats:
                    print(f"  {key}: {stats['mean']:.3f} ± {stats['std']:.3f}")
class WandBLogger(Logger):
    """Weights & Biases logger."""

    def __init__(self, project_name, output_dir=None, exp_name=None, **wandb_kwargs):
        """Initialize WandB logger.

        Args:
            project_name: WandB project name
            output_dir: Output directory
            exp_name: Experiment name
            **wandb_kwargs: Additional WandB arguments
        """
        super().__init__(output_dir, exp_name)

        try:
            import wandb

            self.wandb = wandb
            self.run = wandb.init(
                project=project_name,
                name=self.exp_name,
                dir=self.log_dir,
                **wandb_kwargs,
            )
        except ImportError:
            print("WandB not installed. Using base logger.")
            self.wandb = None

    def log_scalar(self, key, value, step=None):
        """Log scalar to WandB."""
        super().log_scalar(key, value, step)

        if self.wandb is not None:
            self.wandb.log({key: value}, step=step or self.step)

    def log_config(self, config_dict):
        """Log config to WandB."""
        super().log_config(config_dict)

        if self.wandb is not None:
            self.wandb.config.update(config_dict)