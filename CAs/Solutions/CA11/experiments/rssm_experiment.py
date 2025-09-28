"""
RSSM Training Experiment
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import modular components
from world_models.rssm import RecurrentStateSpaceModel
from world_models.trainers import RSSMTrainer
from environments.sequence_environment import SequenceEnvironment
from utils.data_collection import collect_sequence_data, prepare_rssm_batch
from utils.visualization import plot_rssm_training


def run_rssm_experiment(config):
    """Run complete RSSM training experiment"""

    print("=== RSSM Training Experiment ===")
    print(f"Environment: {config['env_name']}")
    print(f"Training steps: {config['train_steps']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Sequence length: {config['sequence_length']}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create environment
    if config['env_name'] == 'sequence':
        env = SequenceEnvironment(memory_size=config['memory_size'])
    else:
        raise ValueError(f"Unknown environment: {config['env_name']}")

    # Collect training data
    print("\nCollecting sequence training data...")
    train_data = collect_sequence_data(env, config['data_collection_episodes'],
                                     config['episode_length'])

    # Create RSSM
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    state_dim = config['state_dim']
    hidden_dim = config['hidden_dim']

    rssm = RecurrentStateSpaceModel(obs_dim, action_dim, state_dim, hidden_dim).to(device)

    # Create trainer
    trainer = RSSMTrainer(rssm, config['learning_rate'], device)

    # Training loop
    print("\nTraining RSSM...")
    pbar = tqdm(range(config['train_steps']), desc="Training")

    for step in pbar:
        # Sample batch of sequences
        batch = prepare_rssm_batch(train_data, config['batch_size'], config['sequence_length'])

        # Move to device
        for key in batch:
            batch[key] = batch[key].to(device)

        # Training step
        losses = trainer.train_step(batch)

        # Update progress bar
        if step % 100 == 0:
            pbar.set_postfix({
                'Total': f"{losses['total']:.4f}",
                'Reconstruction': f"{losses['reconstruction']:.4f}",
                'KL': f"{losses['kl_divergence']:.4f}",
                'Reward': f"{losses['reward']:.4f}"
            })

    print("\nTraining completed!")

    # Analysis and visualization
    print("\nGenerating analysis plots...")
    plot_rssm_training(trainer, f"RSSM Training - {config['env_name']}")

    # Test imagination capability
    print("\nTesting imagination capability...")
    test_imagination(rssm, env, device, config['imagination_steps'])

    return rssm, trainer


def test_imagination(rssm, env, device, imagination_steps):
    """Test RSSM imagination capability"""

    rssm.eval()

    # Start from random state
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, obs_dim]

    # Initialize hidden state
    hidden = torch.zeros(1, rssm.hidden_dim).to(device)

    imagined_states = [obs]
    imagined_rewards = []
    imagined_actions = []

    print(f"Imagining {imagination_steps} steps...")

    for step in range(imagination_steps):
        # Sample random action
        action = env.sample_action()
        action_tensor = torch.FloatTensor(action).unsqueeze(0).unsqueeze(0).to(device)

        # Imagine next state and reward
        with torch.no_grad():
            next_obs_pred, reward_pred, next_hidden = rssm.imagine(obs_tensor, action_tensor, hidden)

        imagined_states.append(next_obs_pred.squeeze(0).squeeze(0).cpu().numpy())
        imagined_rewards.append(reward_pred.squeeze(0).squeeze(0).cpu().numpy())
        imagined_actions.append(action)

        # Update for next step
        obs_tensor = next_obs_pred
        hidden = next_hidden

    # Plot imagination
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    imagined_states = np.array(imagined_states)
    for i in range(min(4, imagined_states.shape[1])):
        plt.plot(imagined_states[:, i], label=f'State {i}')
    plt.title('Imagined State Trajectory')
    plt.xlabel('Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(imagined_rewards, 'g-', linewidth=2)
    plt.title('Imagined Reward Trajectory')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    imagined_actions = np.array(imagined_actions)
    for i in range(min(2, imagined_actions.shape[1])):
        plt.plot(imagined_actions[:, i], label=f'Action {i}')
    plt.title('Action Trajectory')
    plt.xlabel('Step')
    plt.ylabel('Action Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('RSSM Imagination Test', fontsize=16, y=0.98)
    plt.show()

    print(f"Imagination completed: {len(imagined_states)-1} steps, total imagined reward: {sum(imagined_rewards):.2f}")


def compare_real_vs_imagined(rssm, env, device, steps=50):
    """Compare real environment trajectory vs RSSM imagination"""

    rssm.eval()

    # Collect real trajectory
    real_obs_list = []
    real_actions = []
    real_rewards = []

    obs, _ = env.reset()
    real_obs_list.append(obs)

    for _ in range(steps):
        action = env.sample_action()
        next_obs, reward, terminated, truncated, _ = env.step(action)

        real_actions.append(action)
        real_rewards.append(reward)
        real_obs_list.append(next_obs)

        obs = next_obs
        if terminated or truncated:
            break

    # Imagine trajectory with same actions
    imagined_obs_list = []
    imagined_rewards = []

    obs_tensor = torch.FloatTensor(real_obs_list[0]).unsqueeze(0).unsqueeze(0).to(device)
    hidden = torch.zeros(1, rssm.hidden_dim).to(device)
    imagined_obs_list.append(real_obs_list[0])

    for action in real_actions[:len(imagined_obs_list)-1]:
        action_tensor = torch.FloatTensor(action).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            next_obs_pred, reward_pred, next_hidden = rssm.imagine(obs_tensor, action_tensor, hidden)

        imagined_obs_list.append(next_obs_pred.squeeze(0).squeeze(0).cpu().numpy())
        imagined_rewards.append(reward_pred.squeeze(0).squeeze(0).cpu().numpy())

        obs_tensor = next_obs_pred
        hidden = next_hidden

    # Plot comparison
    plt.figure(figsize=(15, 10))

    real_obs_array = np.array(real_obs_list)
    imagined_obs_array = np.array(imagined_obs_list)

    # State comparison
    plt.subplot(2, 2, 1)
    for i in range(min(4, real_obs_array.shape[1])):
        plt.plot(real_obs_array[:, i], 'b-', alpha=0.7, label=f'Real State {i}' if i == 0 else "")
        plt.plot(imagined_obs_array[:, i], 'r--', alpha=0.7, label=f'Imagined State {i}' if i == 0 else "")
    plt.title('Real vs Imagined States')
    plt.xlabel('Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Reward comparison
    plt.subplot(2, 2, 2)
    plt.plot(real_rewards, 'b-', linewidth=2, label='Real Reward')
    plt.plot(imagined_rewards, 'r--', linewidth=2, label='Imagined Reward')
    plt.title('Real vs Imagined Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Prediction error
    plt.subplot(2, 2, 3)
    min_len = min(len(real_obs_array), len(imagined_obs_array))
    prediction_errors = np.mean((real_obs_array[:min_len] - imagined_obs_array[:min_len])**2, axis=1)
    plt.plot(prediction_errors, 'g-', linewidth=2)
    plt.title('State Prediction Error')
    plt.xlabel('Step')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)

    # Reward error
    plt.subplot(2, 2, 4)
    min_reward_len = min(len(real_rewards), len(imagined_rewards))
    reward_errors = (np.array(real_rewards[:min_reward_len]) - np.array(imagined_rewards[:min_reward_len]))**2
    plt.plot(reward_errors, 'orange', linewidth=2)
    plt.title('Reward Prediction Error')
    plt.xlabel('Step')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Real vs Imagined Trajectory Comparison', fontsize=16, y=0.98)
    plt.show()

    print(f"Comparison completed: {min_len} steps")
    print(f"Mean state prediction error: {np.mean(prediction_errors):.4f}")
    print(f"Mean reward prediction error: {np.mean(reward_errors):.4f}")


if __name__ == "__main__":
    # Default configuration
    config = {
        'env_name': 'sequence',
        'state_dim': 32,
        'hidden_dim': 128,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'sequence_length': 50,
        'train_steps': 2000,
        'data_collection_episodes': 100,
        'episode_length': 100,
        'memory_size': 5,
        'imagination_steps': 50
    }

    # Run experiment
    rssm, trainer = run_rssm_experiment(config)

    # Additional comparison
    print("\nRunning real vs imagined comparison...")
    env = SequenceEnvironment(memory_size=config['memory_size'])
    compare_real_vs_imagined(rssm, env, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))