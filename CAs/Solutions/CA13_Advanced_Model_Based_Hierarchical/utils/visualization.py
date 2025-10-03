import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Optional


def plot_training_curves(curves: Dict[str, List[float]], title: str = "Training Curves", 
                        ylabel: str = "Value", save_path: Optional[str] = None):
    """Plot training curves with smoothing"""
    plt.figure(figsize=(12, 6))
    
    for label, series in curves.items():
        if series is None or len(series) == 0:
            continue
        
        series = np.asarray(series)
        if series.size == 0:
            continue
            
        # Smoothing
        window = max(1, len(series) // 50)
        if window > 1:
            kernel = np.ones(window) / window
            smooth = np.convolve(series, kernel, mode='same')
        else:
            smooth = series
        
        plt.plot(series, alpha=0.25, label=f"{label} (raw)")
        plt.plot(smooth, linewidth=2, label=f"{label} (smoothed)")
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Step/Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_bars(labels: List[str], values: List[float], errors: Optional[List[float]] = None,
                title: str = "Comparison", ylabel: str = "Value", save_path: Optional[str] = None):
    """Create bar chart comparison"""
    x = np.arange(len(labels))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, values, yerr=errors, capsize=5, alpha=0.85)
    
    # Color bars based on values
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_latent_trajectories(latents: Optional[np.ndarray], title: str = "Latent Space Trajectories",
                           save_path: Optional[str] = None):
    """Plot trajectories in latent space"""
    if latents is None:
        return
    
    arr = np.array(latents)
    if arr.ndim == 3:  # [T, B, D]
        arr = arr[:, 0]  # Take first batch
    
    if arr.shape[-1] >= 2:
        plt.figure(figsize=(8, 8))
        plt.plot(arr[:, 0], arr[:, 1], '-o', markersize=4, alpha=0.7, linewidth=2)
        plt.scatter(arr[0, 0], arr[0, 1], color='green', s=100, marker='s', label='Start', zorder=5)
        plt.scatter(arr[-1, 0], arr[-1, 1], color='red', s=100, marker='*', label='End', zorder=5)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("z0")
        plt.ylabel("z1")
        plt.legend()
        plt.grid(alpha=0.3)
    else:
        plt.figure(figsize=(12, 4))
        plt.plot(arr[:, 0], '-o', markersize=3, linewidth=2)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Step")
        plt.ylabel("z0")
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_augmentation_examples(original: np.ndarray, augmented_dict: Dict[str, np.ndarray],
                             save_path: Optional[str] = None):
    """Plot data augmentation examples"""
    cols = 1 + len(augmented_dict)
    plt.figure(figsize=(4 * cols, 3))
    
    # Original
    plt.subplot(1, cols, 1)
    plt.plot(original.T, alpha=0.7)
    plt.title("Original", fontweight='bold')
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    
    # Augmented versions
    for i, (name, aug) in enumerate(augmented_dict.items(), start=2):
        plt.subplot(1, cols, i)
        plt.plot(aug.T, alpha=0.7)
        plt.title(name, fontweight='bold')
        plt.xlabel("Feature")
        plt.ylabel("Value")
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_curves_comparison(results: Dict[str, Dict], save_path: Optional[str] = None):
    """Plot learning curves for multiple agents"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training rewards
    ax = axes[0, 0]
    for agent_name, result in results.items():
        rewards = result.get('rewards', [])
        if rewards:
            window = max(1, len(rewards) // 50)
            smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
            ax.plot(smoothed, label=agent_name, linewidth=2)
    
    ax.set_title('Training Rewards', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Episode lengths
    ax = axes[0, 1]
    for agent_name, result in results.items():
        lengths = result.get('lengths', [])
        if lengths:
            window = max(1, len(lengths) // 50)
            smoothed = pd.Series(lengths).rolling(window=window, min_periods=1).mean()
            ax.plot(smoothed, label=agent_name, linewidth=2)
    
    ax.set_title('Episode Lengths', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Length')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Loss curves
    ax = axes[1, 0]
    for agent_name, result in results.items():
        losses = result.get('losses', [])
        if losses:
            ax.plot(losses[:1000], alpha=0.6, label=agent_name)  # Limit for readability
    
    ax.set_title('Training Losses', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Final performance comparison
    ax = axes[1, 1]
    agent_names = []
    final_rewards = []
    
    for agent_name, result in results.items():
        rewards = result.get('rewards', [])
        if rewards:
            agent_names.append(agent_name)
            final_rewards.append(np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards))
    
    if agent_names:
        bars = ax.bar(agent_names, final_rewards, alpha=0.7)
        ax.set_title('Final Performance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Reward (last 20 episodes)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_world_model_analysis(world_model_results: Dict, save_path: Optional[str] = None):
    """Plot world model analysis results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Reconstruction errors
    if 'reconstruction_errors' in world_model_results:
        ax = axes[0, 0]
        errors = world_model_results['reconstruction_errors']
        ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title('Reconstruction Errors', fontweight='bold')
        ax.set_xlabel('MSE')
        ax.set_ylabel('Count')
        ax.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Latent space visualization
    if 'latent_samples' in world_model_results:
        ax = axes[0, 1]
        latent_samples = world_model_results['latent_samples']
        if latent_samples.shape[1] >= 2:
            ax.scatter(latent_samples[:, 0], latent_samples[:, 1], alpha=0.6, s=20)
            ax.set_title('Latent Space Samples', fontweight='bold')
            ax.set_xlabel('z0')
            ax.set_ylabel('z1')
            ax.grid(alpha=0.3)
    
    # Dynamics predictions
    if 'dynamics_errors' in world_model_results:
        ax = axes[0, 2]
        dynamics_errors = world_model_results['dynamics_errors']
        ax.plot(dynamics_errors, alpha=0.7)
        ax.set_title('Dynamics Prediction Errors', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('MSE')
        ax.grid(alpha=0.3)
    
    # Reward predictions
    if 'reward_predictions' in world_model_results:
        ax = axes[1, 0]
        pred_rewards = world_model_results['reward_predictions']
        true_rewards = world_model_results.get('true_rewards', [])
        
        if len(true_rewards) > 0:
            ax.scatter(true_rewards, pred_rewards, alpha=0.6)
            min_val = min(min(true_rewards), min(pred_rewards))
            max_val = max(max(true_rewards), max(pred_rewards))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            ax.set_title('Reward Predictions vs True', fontweight='bold')
            ax.set_xlabel('True Reward')
            ax.set_ylabel('Predicted Reward')
        else:
            ax.hist(pred_rewards, bins=30, alpha=0.7, color='green')
            ax.set_title('Reward Predictions', fontweight='bold')
            ax.set_xlabel('Predicted Reward')
            ax.set_ylabel('Count')
        ax.grid(alpha=0.3)
    
    # Latent trajectory
    if 'latent_trajectory' in world_model_results:
        ax = axes[1, 1]
        trajectory = world_model_results['latent_trajectory']
        if trajectory.shape[1] >= 2:
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', markersize=4, alpha=0.7)
            ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, marker='s', label='Start')
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, marker='*', label='End')
            ax.set_title('Latent Trajectory', fontweight='bold')
            ax.set_xlabel('z0')
            ax.set_ylabel('z1')
            ax.legend()
            ax.grid(alpha=0.3)
    
    # Model components comparison
    ax = axes[1, 2]
    components = ['Encoder', 'Decoder', 'Dynamics', 'Reward']
    losses = [
        world_model_results.get('encoder_loss', 0),
        world_model_results.get('decoder_loss', 0),
        world_model_results.get('dynamics_loss', 0),
        world_model_results.get('reward_loss', 0)
    ]
    
    bars = ax.bar(components, losses, alpha=0.7)
    ax.set_title('Model Component Losses', fontweight='bold')
    ax.set_ylabel('Loss')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_multi_agent_analysis(ma_results: Dict, save_path: Optional[str] = None):
    """Plot multi-agent analysis results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Individual agent rewards
    ax = axes[0, 0]
    for agent_id in range(ma_results.get('n_agents', 3)):
        rewards = ma_results.get(f'agent_{agent_id}_rewards', [])
        if rewards:
            window = max(1, len(rewards) // 50)
            smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
            ax.plot(smoothed, label=f'Agent {agent_id + 1}', linewidth=2)
    
    ax.set_title('Individual Agent Rewards', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Team performance
    ax = axes[0, 1]
    team_rewards = ma_results.get('team_rewards', [])
    if team_rewards:
        window = max(1, len(team_rewards) // 50)
        smoothed = pd.Series(team_rewards).rolling(window=window, min_periods=1).mean()
        ax.plot(team_rewards, alpha=0.3, color='gray')
        ax.plot(smoothed, color='darkgreen', linewidth=2, label='Team Total')
        ax.set_title('Team Performance', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Coordination metric
    ax = axes[0, 2]
    coordination = ma_results.get('coordination_metrics', [])
    if coordination:
        window = max(1, len(coordination) // 50)
        smoothed = pd.Series(coordination).rolling(window=window, min_periods=1).mean()
        ax.plot(coordination, alpha=0.3, color='blue')
        ax.plot(smoothed, color='darkblue', linewidth=2)
        ax.set_title('Coordination Metric', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Coordination Score')
        ax.grid(alpha=0.3)
    
    # Collision frequency
    ax = axes[1, 0]
    collisions = ma_results.get('collision_counts', [])
    if collisions:
        window = max(1, len(collisions) // 50)
        smoothed = pd.Series(collisions).rolling(window=window, min_periods=1).mean()
        ax.plot(collisions, alpha=0.3, color='red')
        ax.plot(smoothed, color='darkred', linewidth=2)
        ax.set_title('Collision Frequency', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Collisions per Episode')
        ax.grid(alpha=0.3)
    
    # Agent correlation
    ax = axes[1, 1]
    n_agents = ma_results.get('n_agents', 3)
    if n_agents > 1:
        # Compute correlation matrix
        agent_rewards = []
        for i in range(n_agents):
            rewards = ma_results.get(f'agent_{i}_rewards', [])
            if rewards:
                agent_rewards.append(rewards[-50:])  # Last 50 episodes
        
        if len(agent_rewards) == n_agents:
            corr_matrix = np.corrcoef(agent_rewards)
            im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
            ax.set_title('Agent Reward Correlation', fontweight='bold')
            ax.set_xticks(range(n_agents))
            ax.set_yticks(range(n_agents))
            ax.set_xticklabels([f'A{i+1}' for i in range(n_agents)])
            ax.set_yticklabels([f'A{i+1}' for i in range(n_agents)])
            
            # Add correlation values
            for i in range(n_agents):
                for j in range(n_agents):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                 ha="center", va="center",
                                 color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
            plt.colorbar(im, ax=ax)
    
    # Final performance
    ax = axes[1, 2]
    final_rewards = []
    agent_labels = []
    for i in range(n_agents):
        rewards = ma_results.get(f'agent_{i}_rewards', [])
        if rewards:
            final_rewards.append(np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards))
            agent_labels.append(f'Agent {i + 1}')
    
    if final_rewards:
        colors = ['red', 'blue', 'gold', 'purple']
        bars = ax.bar(agent_labels, final_rewards, color=colors[:len(final_rewards)], alpha=0.7)
        ax.set_title('Final Performance by Agent', fontweight='bold')
        ax.set_ylabel('Average Reward (last 20 episodes)')
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_summary_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create summary table from results"""
    summary_data = []
    
    for agent_name, result in results.items():
        summary_data.append({
            'Agent': agent_name,
            'Final Reward': np.mean(result.get('rewards', [])[-20:]) if len(result.get('rewards', [])) >= 20 else np.mean(result.get('rewards', [])),
            'Final Std': np.std(result.get('rewards', [])[-20:]) if len(result.get('rewards', [])) >= 20 else np.std(result.get('rewards', [])),
            'Episodes to Solve': len(result.get('rewards', [])),
            'Max Reward': np.max(result.get('rewards', [])),
            'Training Steps': result.get('training_steps', 0)
        })
    
    return pd.DataFrame(summary_data)


def save_results(results: Dict, filepath: str):
    """Save results to file"""
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """Load results from file"""
    import json
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"Results loaded from {filepath}")
    return results