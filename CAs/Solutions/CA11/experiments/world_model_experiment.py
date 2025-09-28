"""
World Model Training Experiment
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import modular components
from world_models.vae import VariationalAutoencoder
from world_models.dynamics import LatentDynamicsModel
from world_models.reward_model import RewardModel
from world_models.world_model import WorldModel
from world_models.trainers import WorldModelTrainer
from environments.continuous_cartpole import ContinuousCartPole
from environments.continuous_pendulum import ContinuousPendulum
from utils.data_collection import collect_world_model_data
from utils.visualization import plot_world_model_training, plot_world_model_analysis


def run_world_model_experiment(config):
    """Run complete world model training experiment"""

    print("=== World Model Training Experiment ===")
    print(f"Environment: {config['env_name']}")
    print(f"Training steps: {config['train_steps']}")
    print(f"Batch size: {config['batch_size']}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create environment
    if config['env_name'] == 'continuous_cartpole':
        env = ContinuousCartPole()
    elif config['env_name'] == 'continuous_pendulum':
        env = ContinuousPendulum()
    else:
        raise ValueError(f"Unknown environment: {config['env_name']}")

    # Collect training data
    print("\nCollecting training data...")
    train_data = collect_world_model_data(env, config['data_collection_steps'],
                                        config['data_collection_episodes'])

    # Create world model components
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_dim = config['latent_dim']

    vae = VariationalAutoencoder(obs_dim, latent_dim, config['vae_hidden_dims']).to(device)
    dynamics = LatentDynamicsModel(latent_dim, action_dim, config['dynamics_hidden_dims'],
                                 stochastic=config['stochastic_dynamics']).to(device)
    reward_model = RewardModel(latent_dim, action_dim, config['reward_hidden_dims']).to(device)

    # Create complete world model
    world_model = WorldModel(vae, dynamics, reward_model).to(device)

    # Create trainer
    trainer = WorldModelTrainer(world_model, config['learning_rate'], device)

    # Training loop
    print("\nTraining world model...")
    pbar = tqdm(range(config['train_steps']), desc="Training")

    for step in pbar:
        # Sample batch
        batch_indices = torch.randperm(len(train_data['observations']))[:config['batch_size']]
        batch = {
            'observations': train_data['observations'][batch_indices].to(device),
            'actions': train_data['actions'][batch_indices].to(device),
            'rewards': train_data['rewards'][batch_indices].to(device),
            'next_observations': train_data['next_observations'][batch_indices].to(device)
        }

        # Training step
        losses = trainer.train_step(batch)

        # Update progress bar
        if step % 100 == 0:
            pbar.set_postfix({
                'VAE Total': f"{losses['vae_total']:.4f}",
                'Dynamics': f"{losses['dynamics']:.4f}",
                'Reward': f"{losses['reward']:.4f}"
            })

    print("\nTraining completed!")

    # Analysis and visualization
    print("\nGenerating analysis plots...")
    plot_world_model_training(trainer, f"World Model Training - {config['env_name']}")
    plot_world_model_analysis(world_model, train_data, device, f"World Model Analysis - {config['env_name']}")

    # Test rollout capability
    print("\nTesting rollout capability...")
    test_rollout(world_model, env, device, config['rollout_steps'])

    return world_model, trainer


def test_rollout(world_model, env, device, rollout_steps):
    """Test world model rollout capability"""

    world_model.eval()

    # Start from random state
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

    rollout_states = [obs]
    rollout_rewards = []
    rollout_actions = []

    print(f"Rolling out {rollout_steps} steps...")

    for step in range(rollout_steps):
        # Sample random action
        action = env.sample_action()
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)

        # Predict next state and reward
        with torch.no_grad():
            next_obs_pred, reward_pred = world_model.predict_next_state_and_reward(obs_tensor, action_tensor)

        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)

        rollout_states.append(next_obs)
        rollout_rewards.append(reward)
        rollout_actions.append(action)

        obs = next_obs
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

        if terminated or truncated:
            break

    # Plot rollout
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    rollout_states = np.array(rollout_states)
    for i in range(min(4, rollout_states.shape[1])):
        plt.plot(rollout_states[:, i], label=f'State {i}')
    plt.title('State Trajectory')
    plt.xlabel('Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(rollout_rewards, 'g-', linewidth=2)
    plt.title('Reward Trajectory')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    rollout_actions = np.array(rollout_actions)
    for i in range(min(2, rollout_actions.shape[1])):
        plt.plot(rollout_actions[:, i], label=f'Action {i}')
    plt.title('Action Trajectory')
    plt.xlabel('Step')
    plt.ylabel('Action Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('World Model Rollout Test', fontsize=16, y=0.98)
    plt.show()

    print(f"Rollout completed: {len(rollout_states)-1} steps, total reward: {sum(rollout_rewards):.2f}")


if __name__ == "__main__":
    # Default configuration
    config = {
        'env_name': 'continuous_cartpole',
        'latent_dim': 32,
        'vae_hidden_dims': [128, 64],
        'dynamics_hidden_dims': [128, 64],
        'reward_hidden_dims': [64, 32],
        'stochastic_dynamics': True,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'train_steps': 2000,
        'data_collection_steps': 10000,
        'data_collection_episodes': 50,
        'rollout_steps': 100
    }

    # Run experiment
    world_model, trainer = run_world_model_experiment(config)