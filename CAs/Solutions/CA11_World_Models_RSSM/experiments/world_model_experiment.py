"""
World Model Experiment Script

This module provides a complete experiment script for training and evaluating
world models with VAE, dynamics, and reward models.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import argparse
import json
from pathlib import Path

from models.vae import VariationalAutoencoder
from models.dynamics import LatentDynamicsModel
from models.reward_model import RewardModel
from models.world_model import WorldModel
from models.trainers import WorldModelTrainer

from environments.continuous_cartpole import ContinuousCartPole
from environments.continuous_pendulum import ContinuousPendulum

from utils.data_collection import collect_world_model_data, create_dataloader
from utils.visualization import plot_world_model_training, plot_reconstruction_comparison


def run_world_model_experiment(
    config: Dict[str, Any],
) -> tuple[WorldModel, Dict[str, Any]]:
    """
    Run a complete world model experiment.

    Args:
        config: Experiment configuration dictionary

    Returns:
        Tuple of (trained_world_model, results)
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

    # Collect data for world model training
    print("Collecting data for world model training...")
    world_model_data = collect_world_model_data(
        env=env,
        steps=config["data_steps"],
        episodes=config.get("data_episodes"),
        seed=config.get("seed", 42),
    )

    print(f"Collected {len(world_model_data['observations'])} transitions")

    # Create data loader
    dataloader = create_dataloader(
        world_model_data,
        batch_size=config["batch_size"],
        device=device,
    )

    # Create world model components
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_dim = config["latent_dim"]

    vae = VariationalAutoencoder(
        obs_dim=obs_dim, latent_dim=latent_dim, hidden_dims=config["vae_hidden_dims"]
    ).to(device)

    dynamics = LatentDynamicsModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dims=config["dynamics_hidden_dims"],
        stochastic=config.get("stochastic_dynamics", True),
    ).to(device)

    reward_model = RewardModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dims=config["reward_hidden_dims"],
    ).to(device)

    world_model = WorldModel(vae, dynamics, reward_model).to(device)

    print(f"World model created:")
    print(f"- VAE: {obs_dim} -> {latent_dim}")
    print(f"- Dynamics: {latent_dim} + {action_dim} -> {latent_dim}")
    print(f"- Reward: {latent_dim} + {action_dim} -> 1")

    # Train world model
    print("Training world model...")
    trainer = WorldModelTrainer(
        world_model=world_model,
        learning_rate=config["learning_rate"],
        device=device,
    )

    losses = trainer.train(
        dataloader=dataloader,
        epochs=config["epochs"],
        beta=config.get("beta_value", 1.0),
        print_interval=config.get("print_interval", 10),
    )

    print("World model training completed!")

    # Evaluate world model
    print("Evaluating world model...")
    world_model.eval()
    
    eval_losses = []
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            obs = batch["observations"]
            actions = batch["actions"]
            next_obs = batch["next_observations"]
            rewards = batch["rewards"]
            
            # Compute loss
            loss_dict = world_model.compute_loss(obs, actions, next_obs, rewards)
            eval_losses.append(loss_dict["total_loss"].item())
            
            # Compute reconstruction error
            _, _, _, z = world_model.vae.encode(obs)
            recon_obs = world_model.vae.decode(z)
            recon_error = torch.mean((obs - recon_obs) ** 2, dim=1)
            reconstruction_errors.extend(recon_error.cpu().numpy())

    avg_eval_loss = np.mean(eval_losses)
    avg_recon_error = np.mean(reconstruction_errors)
    
    print(f"Average evaluation loss: {avg_eval_loss:.4f}")
    print(f"Average reconstruction error: {avg_recon_error:.4f}")

    # Test imagination
    print("Testing imagination...")
    test_batch = next(iter(dataloader))
    test_obs = test_batch["observations"][:5]  # Test on 5 samples
    test_actions = test_batch["actions"][:5]
    
    imagined_trajectory = world_model.imagine_trajectory(
        test_obs, test_actions, horizon=10
    )
    
    print(f"Imagined trajectory shape: {imagined_trajectory['observations'].shape}")

    # Prepare results
    results = {
        "training_losses": losses,
        "eval_loss": avg_eval_loss,
        "reconstruction_error": avg_recon_error,
        "imagined_trajectory": {
            "observations": imagined_trajectory["observations"].cpu().numpy(),
            "rewards": imagined_trajectory["rewards"].cpu().numpy(),
        },
        "config": config,
    }

    # Generate plots
    if config.get("save_plots", True):
        save_dir = Path(config.get("save_dir", "visualizations"))
        save_dir.mkdir(exist_ok=True)

        # Training progress
        plot_world_model_training(
            losses,
            title=f"World Model Training - {env.name}",
            save_path=save_dir / "world_model_training.png",
        )

        # Reconstruction comparison
        test_recon = world_model.vae.decode(world_model.vae.encode(test_obs)[2])
        plot_reconstruction_comparison(
            test_obs.cpu().numpy(),
            test_recon.cpu().numpy(),
            title="Reconstruction Quality",
            save_path=save_dir / "reconstruction_comparison.png",
        )

    # Save model
    if config.get("save_model", True):
        save_dir = Path(config.get("save_dir", "visualizations"))
        save_dir.mkdir(exist_ok=True)

        torch.save(
            {
                "world_model_state_dict": world_model.state_dict(),
                "config": config,
                "results": results,
            },
            save_dir / "world_model.pth",
        )

    return world_model, results


def main():
    """Main function for running world model experiments."""
    parser = argparse.ArgumentParser(description="World Model Experiment")
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--data_steps",
        type=int,
        default=5000,
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
        "vae_hidden_dims": [256, 128],
        "dynamics_hidden_dims": [256, 128],
        "reward_hidden_dims": [128, 64],
        "stochastic_dynamics": True,
        "learning_rate": 1e-3,
        "epochs": args.epochs,
        "batch_size": 64,
        "data_steps": args.data_steps,
        "print_interval": 10,
        "save_plots": True,
        "save_model": True,
        "save_dir": args.save_dir,
        "seed": 42,
        "beta_value": 1.0,
    }

    # Load configuration file if provided
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            file_config = json.load(f)
            config.update(file_config)

    # Run experiment
    world_model, results = run_world_model_experiment(config)

    print("Experiment completed successfully!")
    print(f"Final evaluation loss: {results['eval_loss']:.4f}")
    print(f"Reconstruction error: {results['reconstruction_error']:.4f}")


if __name__ == "__main__":
    main()