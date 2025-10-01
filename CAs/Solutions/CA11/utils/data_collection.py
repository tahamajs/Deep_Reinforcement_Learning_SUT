"""
Data Collection Utilities for World Models

This module provides utilities for collecting experience data from environments
for training world models and model-based RL agents.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Tuple
import gymnasium as gym
from tqdm import tqdm


def collect_world_model_data(
    env: gym.Env,
    steps: int = 1000,
    episodes: int = None,
    random_policy: bool = True,
    seed: int = None
) -> Dict[str, List]:
    """
    Collect experience data for world model training.
    
    Args:
        env: Environment to collect data from
        steps: Total number of steps to collect
        episodes: Number of episodes to collect (if None, use steps)
        random_policy: Whether to use random policy
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing observations, actions, rewards, next_observations
    """
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)
    
    data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'dones': []
    }
    
    obs, _ = env.reset()
    current_steps = 0
    current_episodes = 0
    
    # Determine collection strategy
    if episodes is not None:
        target = episodes
        use_episodes = True
    else:
        target = steps
        use_episodes = False
    
    pbar = tqdm(total=target, desc="Collecting data")
    
    while current_steps < steps and (episodes is None or current_episodes < episodes):
        # Select action
        if random_policy:
            if isinstance(env.action_space, gym.spaces.Box):
                action = env.action_space.sample()
            else:
                action = env.action_space.sample()
        else:
            # Placeholder for custom policy
            action = env.action_space.sample()
        
        # Take step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        data['observations'].append(obs.copy())
        data['actions'].append(action.copy() if isinstance(action, np.ndarray) else [action])
        data['rewards'].append(reward)
        data['next_observations'].append(next_obs.copy())
        data['dones'].append(done)
        
        # Update counters
        current_steps += 1
        if use_episodes:
            pbar.update(1)
        else:
            pbar.update(1)
        
        # Reset if episode ended
        if done:
            obs, _ = env.reset()
            current_episodes += 1
        else:
            obs = next_obs
    
    pbar.close()
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def collect_sequence_data(
    env: gym.Env,
    episodes: int = 50,
    episode_length: int = 20,
    seed: int = None
) -> List[Dict[str, List]]:
    """
    Collect sequence data for temporal world model training.
    
    Args:
        env: Environment to collect data from
        episodes: Number of episodes to collect
        episode_length: Length of each episode
        seed: Random seed for reproducibility
    
    Returns:
        List of episode dictionaries containing observations, actions, rewards
    """
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)
    
    episodes_data = []
    
    for episode in tqdm(range(episodes), desc="Collecting sequence data"):
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': []
        }
        
        obs, _ = env.reset()
        
        for step in range(episode_length):
            # Select action
            if isinstance(env.action_space, gym.spaces.Box):
                action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            
            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            episode_data['observations'].append(obs.copy())
            episode_data['actions'].append(action.copy() if isinstance(action, np.ndarray) else [action])
            episode_data['rewards'].append(reward)
            
            # Update observation
            obs = next_obs
            
            # Break if episode ended early
            if done:
                break
        
        episodes_data.append(episode_data)
    
    return episodes_data


def collect_rollout_data(
    env: gym.Env,
    agent: Any,
    steps: int = 1000,
    episodes: int = None,
    seed: int = None
) -> Dict[str, List]:
    """
    Collect experience data using a specific agent.
    
    Args:
        env: Environment to collect data from
        agent: Agent to use for action selection
        steps: Total number of steps to collect
        episodes: Number of episodes to collect (if None, use steps)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing observations, actions, rewards, next_observations
    """
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)
    
    data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'dones': []
    }
    
    obs, _ = env.reset()
    current_steps = 0
    current_episodes = 0
    
    # Determine collection strategy
    if episodes is not None:
        target = episodes
        use_episodes = True
    else:
        target = steps
        use_episodes = False
    
    pbar = tqdm(total=target, desc="Collecting rollout data")
    
    while current_steps < steps and (episodes is None or current_episodes < episodes):
        # Select action using agent
        if hasattr(agent, 'select_action'):
            action = agent.select_action(obs)
        elif hasattr(agent, 'act'):
            action = agent.act(obs)
        else:
            # Fallback to random action
            action = env.action_space.sample()
        
        # Take step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        data['observations'].append(obs.copy())
        data['actions'].append(action.copy() if isinstance(action, np.ndarray) else [action])
        data['rewards'].append(reward)
        data['next_observations'].append(next_obs.copy())
        data['dones'].append(done)
        
        # Update counters
        current_steps += 1
        if use_episodes:
            pbar.update(1)
        else:
            pbar.update(1)
        
        # Reset if episode ended
        if done:
            obs, _ = env.reset()
            current_episodes += 1
        else:
            obs = next_obs
    
    pbar.close()
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def create_data_loader(
    data: Dict[str, np.ndarray],
    batch_size: int = 64,
    shuffle: bool = True,
    device: torch.device = None
) -> torch.utils.data.DataLoader:
    """
    Create a PyTorch DataLoader from collected data.
    
    Args:
        data: Dictionary containing experience data
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        device: Device to move tensors to
    
    Returns:
        PyTorch DataLoader
    """
    # Convert to tensors
    tensors = {}
    for key, values in data.items():
        if key == 'dones':
            tensors[key] = torch.BoolTensor(values)
        else:
            tensors[key] = torch.FloatTensor(values)
        
        if device is not None:
            tensors[key] = tensors[key].to(device)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        tensors['observations'],
        tensors['actions'],
        tensors['rewards'],
        tensors['next_observations'],
        tensors['dones']
    )
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return dataloader


def split_data(
    data: Dict[str, np.ndarray],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Dictionary containing experience data
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Check ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Get data length
    data_length = len(data['observations'])
    
    # Create indices
    indices = np.arange(data_length)
    np.random.shuffle(indices)
    
    # Split indices
    train_end = int(data_length * train_ratio)
    val_end = int(data_length * (train_ratio + val_ratio))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Split data
    train_data = {key: values[train_indices] for key, values in data.items()}
    val_data = {key: values[val_indices] for key, values in data.items()}
    test_data = {key: values[test_indices] for key, values in data.items()}
    
    return train_data, val_data, test_data


def augment_data(
    data: Dict[str, np.ndarray],
    noise_std: float = 0.01,
    num_augmentations: int = 1
) -> Dict[str, np.ndarray]:
    """
    Augment data by adding noise to observations.
    
    Args:
        data: Dictionary containing experience data
        noise_std: Standard deviation of noise to add
        num_augmentations: Number of augmented copies to create
    
    Returns:
        Augmented data dictionary
    """
    augmented_data = {key: [] for key in data.keys()}
    
    for _ in range(num_augmentations):
        for key, values in data.items():
            if key in ['observations', 'next_observations']:
                # Add noise to observations
                noise = np.random.normal(0, noise_std, values.shape)
                augmented_values = values + noise
            else:
                # Keep other data unchanged
                augmented_values = values.copy()
            
            augmented_data[key].append(augmented_values)
    
    # Concatenate all augmentations
    for key in augmented_data:
        augmented_data[key] = np.concatenate(augmented_data[key], axis=0)
    
    return augmented_data