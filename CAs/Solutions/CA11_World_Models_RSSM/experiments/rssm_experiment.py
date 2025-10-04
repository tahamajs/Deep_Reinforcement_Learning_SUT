"""
RSSM Experiment Script

This module provides a complete experiment script for training and evaluating
Recurrent State Space Models (RSSM) for world modeling.
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

from environments.continuous_cartpole import ContinuousCartPole
from environments.continuous_pendulum import ContinuousPendulum

from utils.data_collection import collect_world_model_data, create_sequence_dataset, create_sequence_dataloader
from utils.visualization import plot_rssm_training, plot_imagination_trajectory


def run_rssm_experiment(
    config: Dict[str, Any],
) -> tuple[RSSM, Dict[str, Any]]:
    """
    Run a complete RSSM experiment.

    Args:
        config: Experiment configuration dictionary

    Returns:
        Tuple of (trained_rssm, results)
    """
    # Set random seeds
    np.random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    if config["env_name"] == "continuous_cartpole":
        env = ContinuousCartPole()
    elif config["env_name"] == "continuous_pendulum":
        env = ContinuousPendulum()
    else:
        raise ValueError(f"Unknown environment: {config['env_name']}")

    print(f"Environment: {env.name}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Collect data for RSSM training
    print("Collecting data for RSSM training...")
    world_model_data = collect_world_model_data(
        env=env,
        steps=config["data_steps"],
        episodes=config.get("data_episodes"),
        seed=config.get("seed", 42),
    )

    print(f"Collected {len(world_model_data['observations'])} transitions")

    # Create sequence dataset
    sequence_data = create_sequence_dataset(
        world_model_data,
        sequence_length=config["sequence_length"],
        overlap=config.get("sequence_overlap", 5),
    )

    print(f"Created {len(sequence_data['observations'])} sequences")

    # Create data loader
    dataloader = create_sequence_dataloader(
        sequence_data,
        batch_size=config["batch_size"],
        device=device,
    )

    # Create RSSM model
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_dim = config["latent_dim"]
    hidden_dim = config["hidden_dim"]
    stochastic_size = config.get("stochastic_size", latent_dim // 2)

    rssm = RSSM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        stochastic_size=stochastic_size,
        rnn_type=config.get("rnn_type", "gru"),
    ).to(device)

    print(f"RSSM created:")
    print(f"- Observation dim: {obs_dim}")
    print(f"- Action dim: {action_dim}")
    print(f"- Latent dim: {latent_dim}")
    print(f"- Hidden dim: {hidden_dim}")
    print(f"- Stochastic size: {stochastic_size}")
    print(f"- RNN type: {config.get('rnn_type', 'gru')}")

    # Train RSSM
    print("Training RSSM...")
    trainer = RSSMTrainer(
        rssm=rssm,
        learning_rate=config["learning_rate"],
        device=device,
    )

    losses = trainer.train(
        dataloader=dataloader,
        epochs=config["epochs"],
        kl_weight=config.get("kl_weight", 0.1),
        print_interval=config.get("print_interval", 10),
    )

    print("RSSM training completed!")

    # Evaluate RSSM
    print("Evaluating RSSM...")
    rssm.eval()
    
    eval_losses = []
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            obs_seq = batch["observations"]
            action_seq = batch["actions"]
            reward_seq = batch["rewards"]
            
            batch_size = obs_seq.shape[0]
            initial_hidden = torch.zeros(batch_size, rssm.deter_dim).to(device)
            
            # Compute loss
            loss_dict = rssm.compute_loss(obs_seq, action_seq, reward_seq, initial_hidden)
            eval_losses.append(loss_dict["total_loss"].item())
            
            # Compute reconstruction error for first observation
            first_obs = obs_seq[:, 0]  # First observation in sequence
            encoded_obs = rssm._encode_observation(first_obs)
            recon_obs = rssm._decode_observation(encoded_obs, initial_hidden)
            recon_error = torch.mean((first_obs - recon_obs) ** 2, dim=1)
            reconstruction_errors.extend(recon_error.cpu().numpy())

    avg_eval_loss = np.mean(eval_losses)
    avg_recon_error = np.mean(reconstruction_errors)
    
    print(f"Average evaluation loss: {avg_eval_loss:.4f}")
    print(f"Average reconstruction error: {avg_recon_error:.4f}")

    # Test imagination
    print("Testing imagination...")
    test_batch = next(iter(dataloader))
    test_obs = test_batch["observations"][0:1]  # Test on 1 sample
    test_actions = test_batch["actions"][0:1]
    
    # Get initial state
    initial_hidden = torch.zeros(1, rssm.deter_dim).to(device)
    initial_latent = rssm._encode_observation(test_obs[:, 0])
    
    # Imagine trajectory
    imagined_obs, imagined_rewards, imagined_hidden = rssm.imagine_trajectory(
        initial_hidden, initial_latent, test_actions
    )
    
    print(f"Imagined trajectory shapes:")
    print(f"- Observations: {imagined_obs.shape}")
    print(f"- Rewards: {imagined_rewards.shape}")
    print(f"- Hidden states: {imagined_hidden.shape}")

    # Prepare results
    results = {
        "training_losses": losses,
        "eval_loss": avg_eval_loss,
        "reconstruction_error": avg_recon_error,
        "imagined_trajectory": {
            "observations": imagined_obs.cpu().numpy(),
            "rewards": imagined_rewards.cpu().numpy(),
            "hidden_states": imagined_hidden.cpu().numpy(),
        },
        "config": config,
    }

    # Generate plots
    if config.get("save_plots", True):
        save_dir = Path(config.get("save_dir", "visualizations"))
        save_dir.mkdir(exist_ok=True)

        # Training progress
        plot_rssm_training(
            losses,
            title=f"RSSM Training - {env.name}",
            save_path=save_dir / "rssm_training.png",
        )

        # Imagination trajectory
        plot_imagination_trajectory(
            imagined_obs[0].cpu().numpy(),
            imagined_rewards[0].cpu().numpy(),
            title="RSSM Imagined Trajectory",
            save_path=save_dir / "rssm_imagination.png",
        )

    # Save model
    if config.get("save_model", True):
        save_dir = Path(config.get("save_dir", "visualizations"))
        save_dir.mkdir(exist_ok=True)

        torch.save(
            {
                "rssm_state_dict": rssm.state_dict(),
                "config": config,
                "results": results,
            },
            save_dir / "rssm_model.pth",
        )

    return rssm, results


def main():
    """Main function for running RSSM experiments."""
    parser = argparse.ArgumentParser(description="RSSM Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="continuous_cartpole",
        choices=["continuous_cartpole", "continuous_pendulum"],
        help="Environment to use",
    )
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--stochastic_size", type=int, default=16, help="Stochastic state size")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=20,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--data_steps",
        type=int,
        default=10000,
        help="Number of data collection steps",
    )
    parser.add_argument(
        "--save_dir", type=str, default="visualizations", help="Directory to save results"
    )

    args = parser.parse_args()

    # Default configuration
    config = {
        "env_name": args.env,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "stochastic_size": args.stochastic_size,
        "rnn_type": "gru",
        "learning_rate": 1e-3,
        "epochs": args.epochs,
        "batch_size": 32,
        "sequence_length": args.sequence_length,
        "sequence_overlap": 5,
        "data_steps": args.data_steps,
        "kl_weight": 0.1,
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
    rssm, results = run_rssm_experiment(config)

    print("Experiment completed successfully!")
    print(f"Final evaluation loss: {results['eval_loss']:.4f}")
    print(f"Reconstruction error: {results['reconstruction_error']:.4f}")


if __name__ == "__main__":
    main()