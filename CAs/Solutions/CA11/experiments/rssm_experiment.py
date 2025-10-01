"""
RSSM Experiment Script

This module provides a complete experiment script for training and evaluating
Recurrent State Space Models on sequence environments.
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

from environments.sequence_environment import SequenceEnvironment

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
    env = SequenceEnvironment(
        memory_size=config.get("memory_size", 5),
        sequence_length=config.get("sequence_length", 20),
    )

    print(f"Environment: {env.name}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Collect data
    print("Collecting sequence data...")
    seq_data = collect_sequence_data(
        env=env,
        episodes=config["data_collection_episodes"],
        episode_length=config.get("sequence_length", 20),
        seed=config.get("seed", 42),
    )

    print(f"Collected {len(seq_data)} episodes")

    # Create RSSM
    obs_dim = env.observation_space.shape[0]
    action_dim = 1  # Discrete actions
    latent_dim = config["latent_dim"]
    hidden_dim = config["hidden_dim"]

    rssm = RSSM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        stochastic_size=config.get("stochastic_size", 32),
        rnn_type=config.get("rnn_type", "gru"),
    ).to(device)

    print(f"RSSM created:")
    print(f"- Observation dim: {obs_dim}")
    print(f"- Action dim: {action_dim}")
    print(f"- Latent dim: {latent_dim}")
    print(f"- Hidden dim: {hidden_dim}")
    print(f"- Stochastic size: {config.get('stochastic_size', 32)}")

    # Create trainer
    trainer = RSSMTrainer(
        rssm=rssm, learning_rate=config["learning_rate"], device=device
    )

    # Training loop
    print("Starting training...")
    best_loss = float("inf")
    patience = config.get("patience", 50)
    patience_counter = 0

    for epoch in range(config["train_epochs"]):
        # Train epoch
        train_losses = trainer.train_epoch(
            data_loader=None,  # We'll use custom training
            num_batches=config.get("batches_per_epoch", 100),
        )

        # Print progress
        if (epoch + 1) % config.get("print_interval", 10) == 0:
            print(f"Epoch {epoch+1}/{config['train_epochs']}:")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")

        # Early stopping
        if train_losses["total_loss"] < best_loss:
            best_loss = train_losses["total_loss"]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training completed!")

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
                "final_loss": best_loss,
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
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "stochastic_size": 32,
        "rnn_type": "gru",
        "learning_rate": args.learning_rate,
        "train_epochs": args.train_epochs,
        "data_collection_episodes": 100,
        "sequence_length": 20,
        "memory_size": 5,
        "batches_per_epoch": 100,
        "patience": 50,
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
