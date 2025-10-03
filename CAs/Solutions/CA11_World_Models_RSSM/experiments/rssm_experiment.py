"""
RSSM Experiment Script

This module provides a complete experiment script for training and evaluating
Recurrent State Space Models (RSSM) on sequence environments.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import argparse
import json
from pathlib import Path

from models.rssm import RSSM
from models.trainers import RSSMTrainer

from environments.sequence_environment import (
    SequenceEnvironment,
    MultiSequenceEnvironment,
)

from utils.data_collection import collect_sequence_data
from utils.visualization import plot_rssm_training


def run_rssm_experiment(config: Dict[str, Any]) -> tuple[RSSM, RSSMTrainer]:
    """
    Run a complete RSSM experiment.

    Args:
        config: Experiment configuration dictionary

    Returns:
        Tuple of (trained_rssm, trainer)
    """
    # Set random seeds
    np.random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    if config["env_name"] == "sequence_environment":
        env = SequenceEnvironment(
            memory_size=config.get("memory_size", 5),
            sequence_length=config.get("sequence_length", 20),
        )
    elif config["env_name"] == "multi_sequence_environment":
        env = MultiSequenceEnvironment(
            num_sequences=config.get("num_sequences", 3),
            sequence_length=config.get("sequence_length", 15),
        )
    else:
        raise ValueError(f"Unknown environment: {config['env_name']}")

    print(f"Environment: {env.name}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Collect data
    print("Collecting training data...")
    data = collect_sequence_data(
        env=env,
        episodes=config["data_collection_episodes"],
        episode_length=config.get("episode_length", 20),
        seed=config.get("seed", 42),
    )

    print(f"Collected {len(data)} episodes")

    # Split data
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create RSSM
    obs_dim = env.observation_space.shape[0]
    action_dim = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else env.action_space.shape[0]
    )

    rssm = RSSM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        stochastic_size=config.get("stochastic_size", 32),
        rnn_type=config.get("rnn_type", "gru"),
    ).to(device)

    print(f"RSSM created:")
    print(f"- Observation dim: {obs_dim}")
    print(f"- Action dim: {action_dim}")
    print(f"- Latent dim: {config['latent_dim']}")
    print(f"- Hidden dim: {config['hidden_dim']}")
    print(f"- Stochastic size: {config.get('stochastic_size', 32)}")

    # Create trainer
    trainer = RSSMTrainer(
        rssm=rssm, learning_rate=config["learning_rate"], device=device
    )

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    patience = config.get("patience", 30)
    patience_counter = 0

    for epoch in range(config["train_epochs"]):
        # Train epoch
        train_losses = trainer.train_epoch(
            train_data, num_batches=config.get("batches_per_epoch", 50)
        )

        # Validate
        val_losses = trainer.evaluate(
            val_data, num_batches=config.get("val_batches", 20)
        )

        # Print progress
        if (epoch + 1) % config.get("print_interval", 10) == 0:
            print(f"Epoch {epoch+1}/{config['train_epochs']}:")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Val Loss: {val_losses['total_loss']:.4f}")

        # Early stopping
        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training completed!")

    # Final evaluation
    test_losses = trainer.evaluate(
        test_data, num_batches=config.get("test_batches", 20)
    )
    print(f"Final test loss: {test_losses['total_loss']:.4f}")

    # Generate plots
    if config.get("save_plots", True):
        save_dir = Path(config.get("save_dir", "results"))
        save_dir.mkdir(exist_ok=True)

        # Training progress
        plot_rssm_training(
            trainer,
            title=f"RSSM Training - {env.name}",
            save_path=save_dir / "rssm_training_progress.png",
        )

    # Save model
    if config.get("save_model", True):
        save_dir = Path(config.get("save_dir", "results"))
        save_dir.mkdir(exist_ok=True)

        torch.save(
            {
                "rssm_state_dict": rssm.state_dict(),
                "config": config,
                "test_losses": test_losses,
            },
            save_dir / "rssm_model.pth",
        )

    return rssm, trainer


def main():
    """Main function for running RSSM experiments."""
    parser = argparse.ArgumentParser(description="RSSM Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rssm_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="sequence_environment",
        choices=["sequence_environment", "multi_sequence_environment"],
        help="Environment to use",
    )
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument(
        "--train_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Directory to save results"
    )

    args = parser.parse_args()

    # Default configuration
    config = {
        "env_name": args.env,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "stochastic_size": 32,
        "rnn_type": "gru",
        "learning_rate": args.learning_rate,
        "train_epochs": args.train_epochs,
        "data_collection_episodes": 100,
        "episode_length": 20,
        "memory_size": 5,
        "num_sequences": 3,
        "sequence_length": 15,
        "batches_per_epoch": 50,
        "val_batches": 20,
        "test_batches": 20,
        "patience": 30,
        "print_interval": 10,
        "save_plots": True,
        "save_model": True,
        "save_dir": args.save_dir,
        "seed": 42,
    }

    # Load configuration file if provided
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            file_config = json.load(f)
            config.update(file_config)

    # Run experiment
    rssm, trainer = run_rssm_experiment(config)

    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()
