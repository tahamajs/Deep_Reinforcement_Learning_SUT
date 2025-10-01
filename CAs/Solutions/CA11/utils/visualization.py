"""
Visualization Utilities for World Models

This module provides visualization utilities for world models, including
training progress, model predictions, and analysis plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd


def plot_world_model_training(
    trainer: Any,
    title: str = "World Model Training Progress",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot world model training progress.
    
    Args:
        trainer: World model trainer with loss history
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot total loss
    axes[0, 0].plot(trainer.loss_history['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot VAE loss
    axes[0, 1].plot(trainer.loss_history['vae_loss'], 'g-', linewidth=2)
    axes[0, 1].set_title('VAE Loss')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot dynamics loss
    axes[1, 0].plot(trainer.loss_history['dynamics_loss'], 'r-', linewidth=2)
    axes[1, 0].set_title('Dynamics Loss')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot reward loss
    axes[1, 1].plot(trainer.loss_history['reward_loss'], 'purple', linewidth=2)
    axes[1, 1].set_title('Reward Loss')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_rssm_training(
    trainer: Any,
    title: str = "RSSM Training Progress",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot RSSM training progress.
    
    Args:
        trainer: RSSM trainer with loss history
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot total loss
    axes[0, 0].plot(trainer.loss_history['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot reconstruction loss
    axes[0, 1].plot(trainer.loss_history['reconstruction_loss'], 'g-', linewidth=2)
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot reward loss
    axes[1, 0].plot(trainer.loss_history['reward_loss'], 'r-', linewidth=2)
    axes[1, 0].set_title('Reward Loss')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot KL loss
    axes[1, 1].plot(trainer.loss_history['kl_loss'], 'purple', linewidth=2)
    axes[1, 1].set_title('KL Divergence Loss')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_world_model_predictions(
    world_model: Any,
    test_data: Dict[str, np.ndarray],
    num_samples: int = 5,
    title: str = "World Model Predictions",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot world model predictions vs ground truth.
    
    Args:
        world_model: Trained world model
        test_data: Test data dictionary
        num_samples: Number of samples to plot
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    world_model.eval()
    
    # Select random samples
    indices = np.random.choice(len(test_data['observations']), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            obs = torch.FloatTensor(test_data['observations'][idx]).unsqueeze(0)
            action = torch.FloatTensor(test_data['actions'][idx]).unsqueeze(0)
            true_next_obs = test_data['next_observations'][idx]
            true_reward = test_data['rewards'][idx]
            
            # Get predictions
            pred_next_obs, pred_reward = world_model.predict_next_state_and_reward(obs, action)
            pred_next_obs = pred_next_obs.squeeze(0).cpu().numpy()
            pred_reward = pred_reward.item()
            
            # Plot observations
            obs_dim = len(obs.squeeze(0))
            x = np.arange(obs_dim)
            
            axes[i, 0].bar(x - 0.2, obs.squeeze(0), width=0.4, label='Current', alpha=0.7)
            axes[i, 0].bar(x + 0.2, true_next_obs, width=0.4, label='True Next', alpha=0.7)
            axes[i, 0].set_title(f'Sample {i+1}: Observations')
            axes[i, 0].set_xlabel('Dimension')
            axes[i, 0].set_ylabel('Value')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot predictions
            axes[i, 1].bar(x - 0.2, obs.squeeze(0), width=0.4, label='Current', alpha=0.7)
            axes[i, 1].bar(x + 0.2, pred_next_obs, width=0.4, label='Predicted Next', alpha=0.7)
            axes[i, 1].set_title(f'Sample {i+1}: Predictions')
            axes[i, 1].set_xlabel('Dimension')
            axes[i, 1].set_ylabel('Value')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
            
            # Plot rewards
            axes[i, 2].bar(['True', 'Predicted'], [true_reward, pred_reward], 
                          color=['blue', 'red'], alpha=0.7)
            axes[i, 2].set_title(f'Sample {i+1}: Rewards')
            axes[i, 2].set_ylabel('Reward')
            axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_trajectory_rollout(
    world_model: Any,
    initial_obs: np.ndarray,
    actions: np.ndarray,
    true_trajectory: Optional[np.ndarray] = None,
    title: str = "Trajectory Rollout",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot trajectory rollout from world model.
    
    Args:
        world_model: Trained world model
        initial_obs: Initial observation
        actions: Sequence of actions
        true_trajectory: True trajectory (optional)
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    world_model.eval()
    
    # Generate rollout
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(initial_obs).unsqueeze(0)
        actions_tensor = torch.FloatTensor(actions).unsqueeze(0)
        
        trajectory = world_model.imagine_trajectory(obs_tensor, actions_tensor, horizon=len(actions))
        
        pred_obs = trajectory['observations'].squeeze(0).cpu().numpy()
        pred_rewards = trajectory['rewards'].squeeze(0).cpu().numpy()
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot observations over time
    obs_dim = pred_obs.shape[1]
    time_steps = np.arange(len(pred_obs))
    
    for i in range(min(4, obs_dim)):
        axes[0, 0].plot(time_steps, pred_obs[:, i], label=f'Dim {i}', linewidth=2)
    
    if true_trajectory is not None:
        for i in range(min(4, true_trajectory.shape[1])):
            axes[0, 0].plot(time_steps, true_trajectory[:, i], '--', 
                           label=f'True Dim {i}', linewidth=2, alpha=0.7)
    
    axes[0, 0].set_title('Observations Over Time')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot rewards
    axes[0, 1].plot(time_steps[1:], pred_rewards, 'g-o', linewidth=2, markersize=4)
    axes[0, 1].set_title('Predicted Rewards')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot observation heatmap
    im = axes[1, 0].imshow(pred_obs.T, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Observation Heatmap')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Observation Dimension')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot action sequence
    if len(actions.shape) > 1:
        for i in range(actions.shape[1]):
            axes[1, 1].plot(time_steps[1:], actions[:, i], label=f'Action {i}', linewidth=2)
    else:
        axes[1, 1].plot(time_steps[1:], actions, 'r-o', linewidth=2, markersize=4)
    
    axes[1, 1].set_title('Action Sequence')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Action Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_latent_space_analysis(
    world_model: Any,
    data: Dict[str, np.ndarray],
    title: str = "Latent Space Analysis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot latent space analysis for world model.
    
    Args:
        world_model: Trained world model
        data: Data dictionary
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    world_model.eval()
    
    # Encode observations to latent space
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(data['observations'])
        latents = world_model.encode_observations(obs_tensor).cpu().numpy()
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot latent space distribution
    latent_dim = latents.shape[1]
    for i in range(min(4, latent_dim)):
        axes[0, 0].hist(latents[:, i], bins=50, alpha=0.7, label=f'Dim {i}')
    
    axes[0, 0].set_title('Latent Space Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot latent space correlation
    if latent_dim >= 2:
        axes[0, 1].scatter(latents[:, 0], latents[:, 1], alpha=0.5, s=1)
        axes[0, 1].set_title('Latent Space Correlation (Dim 0 vs Dim 1)')
        axes[0, 1].set_xlabel('Latent Dim 0')
        axes[0, 1].set_ylabel('Latent Dim 1')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot latent space over time
    if len(latents) > 100:
        sample_indices = np.random.choice(len(latents), 100, replace=False)
        sample_latents = latents[sample_indices]
        time_steps = np.arange(len(sample_latents))
        
        for i in range(min(4, latent_dim)):
            axes[1, 0].plot(time_steps, sample_latents[:, i], label=f'Dim {i}', linewidth=1)
        
        axes[1, 0].set_title('Latent Space Over Time (Sample)')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot latent space heatmap
    if latent_dim >= 2:
        im = axes[1, 1].imshow(latents[:100].T, aspect='auto', cmap='viridis')
        axes[1, 1].set_title('Latent Space Heatmap (First 100 samples)')
        axes[1, 1].set_xlabel('Sample')
        axes[1, 1].set_ylabel('Latent Dimension')
        plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dreamer_training(
    dreamer_agent: Any,
    title: str = "Dreamer Agent Training Progress",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Dreamer agent training progress.
    
    Args:
        dreamer_agent: Dreamer agent with training statistics
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot actor loss
    axes[0, 0].plot(dreamer_agent.stats['actor_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Actor Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot critic loss
    axes[0, 1].plot(dreamer_agent.stats['critic_loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('Critic Loss')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot imagination reward
    axes[1, 0].plot(dreamer_agent.stats['imagination_reward'], 'g-', linewidth=2)
    axes[1, 0].set_title('Imagination Reward')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot policy entropy
    axes[1, 1].plot(dreamer_agent.stats['policy_entropy'], 'purple', linewidth=2)
    axes[1, 1].set_title('Policy Entropy')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Model Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison metrics for different models.
    
    Args:
        metrics: Dictionary of metrics for different models
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot training loss comparison
    for model_name, loss_history in metrics.items():
        if 'loss' in model_name.lower():
            axes[0, 0].plot(loss_history, label=model_name, linewidth=2)
    
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot reward comparison
    for model_name, reward_history in metrics.items():
        if 'reward' in model_name.lower():
            axes[0, 1].plot(reward_history, label=model_name, linewidth=2)
    
    axes[0, 1].set_title('Reward Comparison')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot accuracy comparison
    for model_name, accuracy_history in metrics.items():
        if 'accuracy' in model_name.lower():
            axes[1, 0].plot(accuracy_history, label=model_name, linewidth=2)
    
    axes[1, 0].set_title('Accuracy Comparison')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot final performance comparison
    final_metrics = {}
    for model_name, history in metrics.items():
        if len(history) > 0:
            final_metrics[model_name] = history[-1]
    
    if final_metrics:
        model_names = list(final_metrics.keys())
        values = list(final_metrics.values())
        axes[1, 1].bar(model_names, values, alpha=0.7)
        axes[1, 1].set_title('Final Performance Comparison')
        axes[1, 1].set_ylabel('Final Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig