"""
World Model Experiment Script

This module provides a complete experiment script for training and evaluating
world models on various environments.
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

from utils.data_collection import collect_world_model_data, split_data, create_data_loader
from utils.visualization import plot_world_model_training, plot_world_model_predictions


def run_world_model_experiment(config: Dict[str, Any]) -> tuple[WorldModel, WorldModelTrainer]:
    """
    Run a complete world model experiment.
    
    Args:
        config: Experiment configuration dictionary
    
    Returns:
        Tuple of (trained_world_model, trainer)
    """
    # Set random seeds
    np.random.seed(config.get('seed', 42))
    torch.manual_seed(config.get('seed', 42))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create environment
    if config['env_name'] == 'continuous_cartpole':
        env = ContinuousCartPole()
    elif config['env_name'] == 'continuous_pendulum':
        env = ContinuousPendulum()
    else:
        raise ValueError(f"Unknown environment: {config['env_name']}")

    print(f"Environment: {env.name}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    # Collect data
    print("Collecting training data...")
    data = collect_world_model_data(
        env=env,
        steps=config['data_collection_steps'],
        episodes=config.get('data_collection_episodes'),
        seed=config.get('seed', 42)
    )
    
    print(f"Collected {len(data['observations'])} transitions")
    
    # Split data
    train_data, val_data, test_data = split_data(
        data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=config.get('seed', 42)
    )
    
    print(f"Train: {len(train_data['observations'])}, "
          f"Val: {len(val_data['observations'])}, "
          f"Test: {len(test_data['observations'])}")
    
    # Create data loaders
    train_loader = create_data_loader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        device=device
    )
    
    val_loader = create_data_loader(
        val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        device=device
    )
    
    # Create world model components
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_dim = config['latent_dim']
    
    vae = VariationalAutoencoder(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dims=config['vae_hidden_dims']
    ).to(device)
    
    dynamics = LatentDynamicsModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dims=config['dynamics_hidden_dims'],
        stochastic=config.get('stochastic_dynamics', True)
    ).to(device)
    
    reward_model = RewardModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dims=config['reward_hidden_dims']
    ).to(device)

    world_model = WorldModel(vae, dynamics, reward_model).to(device)

    print(f"World model created:")
    print(f"- VAE: {obs_dim} -> {latent_dim}")
    print(f"- Dynamics: {latent_dim} + {action_dim} -> {latent_dim}")
    print(f"- Reward: {latent_dim} + {action_dim} -> 1")
    
    # Create trainer
    trainer = WorldModelTrainer(
        world_model=world_model,
        learning_rate=config['learning_rate'],
        device=device,
        beta_schedule=config.get('beta_schedule', 'constant'),
        beta_value=config.get('beta_value', 1.0)
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience = config.get('patience', 50)
    patience_counter = 0
    
    for epoch in range(config['train_epochs']):
        # Train epoch
        train_losses = trainer.train_epoch(train_loader)
        
        # Validate
        val_losses = trainer.evaluate(val_loader)
        
        # Print progress
        if (epoch + 1) % config.get('print_interval', 10) == 0:
            print(f"Epoch {epoch+1}/{config['train_epochs']}:")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Val Loss: {val_losses['total_loss']:.4f}")
        
        # Early stopping
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training completed!")
    
    # Final evaluation
    test_loader = create_data_loader(
        test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        device=device
    )
    
    test_losses = trainer.evaluate(test_loader)
    print(f"Final test loss: {test_losses['total_loss']:.4f}")
    
    # Generate plots
    if config.get('save_plots', True):
        save_dir = Path(config.get('save_dir', 'results'))
        save_dir.mkdir(exist_ok=True)
        
        # Training progress
        plot_world_model_training(
            trainer,
            title=f"World Model Training - {env.name}",
            save_path=save_dir / "training_progress.png"
        )
        
        # Predictions
        plot_world_model_predictions(
            world_model,
            test_data,
            num_samples=5,
            title=f"World Model Predictions - {env.name}",
            save_path=save_dir / "predictions.png"
        )
    
    # Save model
    if config.get('save_model', True):
        save_dir = Path(config.get('save_dir', 'results'))
        save_dir.mkdir(exist_ok=True)
        
        torch.save({
            'world_model_state_dict': world_model.state_dict(),
            'config': config,
            'test_losses': test_losses
        }, save_dir / "world_model.pth")
    
    return world_model, trainer


def main():
    """Main function for running world model experiments."""
    parser = argparse.ArgumentParser(description='World Model Experiment')
    parser.add_argument('--config', type=str, default='configs/world_model_config.json',
                       help='Path to configuration file')
    parser.add_argument('--env', type=str, default='continuous_cartpole',
                       choices=['continuous_cartpole', 'continuous_pendulum'],
                       help='Environment to use')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension')
    parser.add_argument('--train_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'env_name': args.env,
        'latent_dim': args.latent_dim,
        'vae_hidden_dims': [256, 128],
        'dynamics_hidden_dims': [256, 128],
        'reward_hidden_dims': [128, 64],
        'stochastic_dynamics': True,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'train_epochs': args.train_epochs,
        'data_collection_steps': 5000,
        'data_collection_episodes': 20,
        'patience': 50,
        'print_interval': 10,
        'save_plots': True,
        'save_model': True,
        'save_dir': args.save_dir,
        'seed': 42,
        'beta_schedule': 'constant',
        'beta_value': 1.0
    }
    
    # Load configuration file if provided
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Run experiment
    world_model, trainer = run_world_model_experiment(config)

    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()