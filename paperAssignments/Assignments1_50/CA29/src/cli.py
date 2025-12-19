"""Command-line interface for running SAC experiments."""

import argparse
from pathlib import Path

from .config import load_config, SACConfig
from .experiment import Experiment


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Run Soft Actor-Critic (SAC) experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the random seed from config."
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Override the environment name from config."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Override the log directory from config."
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(str(config_path))

    # Apply overrides
    if args.seed is not None:
        config.seed = args.seed
    if args.env is not None:
        config.env_name = args.env
    if args.log_dir is not None:
        config.log_dir = args.log_dir

    # Run experiment
    print(f"Starting SAC experiment with config: {config}")
    exp = Experiment(config)
    exp.run_experiment()
    print("Experiment completed.")


if __name__ == "__main__":
    main()