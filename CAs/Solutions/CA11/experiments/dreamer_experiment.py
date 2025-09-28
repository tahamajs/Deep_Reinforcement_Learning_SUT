"""
Dreamer Agent Training Experiment
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
from agents.latent_actor import LatentActor
from agents.latent_critic import LatentCritic
from agents.dreamer_agent import DreamerAgent
from environments.continuous_cartpole import ContinuousCartPole
from environments.continuous_pendulum import ContinuousPendulum
from utils.data_collection import collect_world_model_data
from utils.visualization import plot_world_model_training, plot_dreamer_training, plot_performance_comparison


def run_dreamer_experiment(config):
    """Run complete Dreamer agent training experiment"""

    print("=== Dreamer Agent Training Experiment ===")
    print(f"Environment: {config['env_name']}")
    print(f"Training episodes: {config['train_episodes']}")
    print(f"World model pre-training steps: {config['wm_pretrain_steps']}")

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

    # Collect training data for world model
    print("\nCollecting training data for world model...")
    train_data = collect_world_model_data(env, config['data_collection_steps'],
                                        config['data_collection_episodes'])

    # Create and pre-train world model
    print("\nPre-training world model...")
    world_model = create_and_pretrain_world_model(env, train_data, config, device)

    # Create Dreamer agent
    print("\nCreating Dreamer agent...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_dim = config['latent_dim']

    actor = LatentActor(latent_dim, action_dim, config['actor_hidden_dims']).to(device)
    critic = LatentCritic(latent_dim, config['critic_hidden_dims']).to(device)

    dreamer_agent = DreamerAgent(
        world_model=world_model,
        actor=actor,
        critic=critic,
        imagination_horizon=config['imagination_horizon'],
        gamma=config['gamma'],
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        device=device
    )

    # Training loop
    print("\nTraining Dreamer agent...")
    training_rewards = []
    episode_lengths = []

    for episode in tqdm(range(config['train_episodes']), desc="Training Episodes"):
        episode_reward = 0
        episode_length = 0

        # Collect episode data
        obs, _ = env.reset()
        obs_tensor = torch.FloatTensor(obs).to(device)

        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': []
        }

        done = False
        while not done and episode_length < config['max_episode_length']:
            # Agent selects action
            with torch.no_grad():
                action = dreamer_agent.act(obs_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()

            # Environment step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['next_observations'].append(next_obs)

            obs = next_obs
            obs_tensor = torch.FloatTensor(obs).to(device)
            episode_reward += reward
            episode_length += 1

        # Convert to tensors
        episode_tensors = {
            'observations': torch.FloatTensor(np.array(episode_data['observations'])).to(device),
            'actions': torch.FloatTensor(np.array(episode_data['actions'])).to(device),
            'rewards': torch.FloatTensor(np.array(episode_data['rewards'])).to(device),
            'next_observations': torch.FloatTensor(np.array(episode_data['next_observations'])).to(device)
        }

        # Update agent
        dreamer_agent.update(episode_tensors)

        training_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(training_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            print(f"Episode {episode+1}: Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")

    print("\nTraining completed!")

    # Evaluation
    print("\nEvaluating trained agent...")
    eval_rewards = evaluate_agent(dreamer_agent, env, config['eval_episodes'], device)

    # Comparison with random policy
    print("\nComparing with random policy...")
    random_rewards = evaluate_random_policy(env, config['eval_episodes'])

    # Analysis and visualization
    print("\nGenerating analysis plots...")
    plot_dreamer_training(dreamer_agent, training_rewards, f"Dreamer Training - {config['env_name']}")
    plot_performance_comparison(eval_rewards, random_rewards, f"Performance Comparison - {config['env_name']}")

    return dreamer_agent, training_rewards, eval_rewards, random_rewards


def create_and_pretrain_world_model(env, train_data, config, device):
    """Create and pre-train world model"""

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_dim = config['latent_dim']

    # Create components
    vae = VariationalAutoencoder(obs_dim, latent_dim, config['vae_hidden_dims']).to(device)
    dynamics = LatentDynamicsModel(latent_dim, action_dim, config['dynamics_hidden_dims'],
                                 stochastic=config['stochastic_dynamics']).to(device)
    reward_model = RewardModel(latent_dim, action_dim, config['reward_hidden_dims']).to(device)

    world_model = WorldModel(vae, dynamics, reward_model).to(device)
    trainer = WorldModelTrainer(world_model, config['wm_learning_rate'], device)

    # Pre-training
    print(f"Pre-training world model for {config['wm_pretrain_steps']} steps...")
    for step in tqdm(range(config['wm_pretrain_steps']), desc="Pre-training"):
        batch_indices = torch.randperm(len(train_data['observations']))[:config['wm_batch_size']]
        batch = {
            'observations': train_data['observations'][batch_indices].to(device),
            'actions': train_data['actions'][batch_indices].to(device),
            'rewards': train_data['rewards'][batch_indices].to(device),
            'next_observations': train_data['next_observations'][batch_indices].to(device)
        }

        trainer.train_step(batch)

    return world_model


def evaluate_agent(agent, env, num_episodes, device):
    """Evaluate trained agent"""

    agent.eval()
    eval_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs_tensor = torch.FloatTensor(obs).to(device)
        episode_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 1000:  # Prevent infinite episodes
            with torch.no_grad():
                action = agent.act(obs_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_tensor = torch.FloatTensor(next_obs).to(device)
            episode_reward += reward
            step_count += 1

        eval_rewards.append(episode_reward)

    return eval_rewards


def evaluate_random_policy(env, num_episodes):
    """Evaluate random policy for comparison"""

    random_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 1000:
            action = env.sample_action()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step_count += 1

        random_rewards.append(episode_reward)

    return random_rewards


if __name__ == "__main__":
    # Default configuration
    config = {
        'env_name': 'continuous_cartpole',
        'latent_dim': 32,
        'vae_hidden_dims': [128, 64],
        'dynamics_hidden_dims': [128, 64],
        'reward_hidden_dims': [64, 32],
        'actor_hidden_dims': [128, 64],
        'critic_hidden_dims': [128, 64],
        'stochastic_dynamics': True,
        'wm_learning_rate': 1e-3,
        'actor_lr': 1e-4,
        'critic_lr': 1e-4,
        'wm_pretrain_steps': 1000,
        'wm_batch_size': 64,
        'train_episodes': 500,
        'max_episode_length': 200,
        'imagination_horizon': 15,
        'gamma': 0.99,
        'data_collection_steps': 5000,
        'data_collection_episodes': 20,
        'eval_episodes': 20
    }

    # Run experiment
    dreamer_agent, training_rewards, eval_rewards, random_rewards = run_dreamer_experiment(config)

    # Print final results
    print("\n=== Final Results ===")
    print(f"Dreamer Agent - Mean: {np.mean(eval_rewards):.2f}, Std: {np.std(eval_rewards):.2f}")
    print(f"Random Policy - Mean: {np.mean(random_rewards):.2f}, Std: {np.std(random_rewards):.2f}")
    print(f"Improvement: {((np.mean(eval_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100):.1f}%")