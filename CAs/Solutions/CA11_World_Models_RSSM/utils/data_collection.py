"""
Data Collection Utilities for World Models

This module provides utilities for collecting data from environments
for training world models and reinforcement learning agents.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import random
from collections import deque


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_world_model_data(
    env,
    steps: int = 1000,
    episodes: Optional[int] = None,
    seed: int = 42,
    action_noise: float = 0.1,
    random_action_prob: float = 0.5,
) -> Dict[str, List[np.ndarray]]:
    """
    Collect data from environment for world model training
    
    Args:
        env: Gymnasium environment
        steps: Number of steps to collect
        episodes: Number of episodes to collect (overrides steps if provided)
        seed: Random seed
        action_noise: Noise level for actions
        random_action_prob: Probability of taking random action
    
    Returns:
        Dictionary containing observations, actions, next_observations, rewards
    """
    set_seed(seed)
    
    observations = []
    actions = []
    next_observations = []
    rewards = []
    dones = []
    
    obs, _ = env.reset(seed=seed)
    current_step = 0
    episode_count = 0
    
    while True:
        if episodes is not None and episode_count >= episodes:
            break
        if episodes is None and current_step >= steps:
            break
            
        # Select action
        if random.random() < random_action_prob:
            # Random action
                action = env.action_space.sample()
        else:
            # Add noise to previous action or use zero
            if len(actions) > 0:
                action = np.clip(
                    actions[-1] + np.random.normal(0, action_noise, size=env.action_space.shape),
                    env.action_space.low,
                    env.action_space.high
                )
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
        
        # Take step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        observations.append(obs.copy())
        actions.append(action.copy())
        next_observations.append(next_obs.copy())
        rewards.append(reward)
        dones.append(done)
        
        obs = next_obs
        current_step += 1
        
        if done:
            obs, _ = env.reset()
            episode_count += 1
    
    return {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": rewards,
        "dones": dones,
    }


def collect_rollout_data(
    env,
    agent,
    num_episodes: int = 10,
    max_steps: int = 200,
    seed: int = 42,
    deterministic: bool = False,
) -> Dict[str, List[np.ndarray]]:
    """
    Collect rollout data using a trained agent
    
    Args:
        env: Gymnasium environment
        agent: Trained agent
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        seed: Random seed
        deterministic: Whether to use deterministic actions
    
    Returns:
        Dictionary containing rollout data
    """
    set_seed(seed)
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_dones = []
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Get action from agent
            if hasattr(agent, 'select_action'):
                action = agent.select_action(obs, deterministic=deterministic)
            elif hasattr(agent, 'act'):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = agent.act(obs_tensor, deterministic=deterministic)
                action = action.cpu().numpy()
            else:
                # Fallback to random action
                action = env.action_space.sample()
            
            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            all_observations.append(obs.copy())
            all_actions.append(action.copy())
            all_rewards.append(reward)
            all_dones.append(done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        "observations": all_observations,
        "actions": all_actions,
        "rewards": all_rewards,
        "dones": all_dones,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def create_sequence_dataset(
    data: Dict[str, List[np.ndarray]],
    sequence_length: int = 10,
    overlap: int = 5,
) -> Dict[str, List[np.ndarray]]:
    """
    Create sequence dataset from collected data
    
    Args:
        data: Dictionary containing transitions
        sequence_length: Length of sequences
        overlap: Overlap between sequences
    
    Returns:
        Dictionary containing sequences
    """
    observations = data["observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    
    seq_observations = []
    seq_actions = []
    seq_rewards = []
    
    step_size = max(1, sequence_length - overlap)
    
    for i in range(0, len(observations) - sequence_length + 1, step_size):
        seq_obs = observations[i:i + sequence_length]
        seq_act = actions[i:i + sequence_length]
        seq_rew = rewards[i:i + sequence_length]
        
        seq_observations.append(np.array(seq_obs))
        seq_actions.append(np.array(seq_act))
        seq_rewards.append(np.array(seq_rew))
    
    return {
        "observations": seq_observations,
        "actions": seq_actions,
        "rewards": seq_rewards,
    }


def create_dataloader(
    data: Dict[str, List[np.ndarray]],
    batch_size: int = 32,
    shuffle: bool = True,
    device: torch.device = None,
) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoader from collected data
    
    Args:
        data: Dictionary containing data arrays
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        device: Device to move tensors to
    
    Returns:
        PyTorch DataLoader
    """
    device = device or get_device()
    
    # Convert to tensors
    dataset = {
        "observations": torch.FloatTensor(np.array(data["observations"])),
        "actions": torch.FloatTensor(np.array(data["actions"])),
        "next_observations": torch.FloatTensor(np.array(data["next_observations"])),
        "rewards": torch.FloatTensor(np.array(data["rewards"])),
    }
    
    # Create dataset class
    class TransitionDataset(torch.utils.data.Dataset):
        def __init__(self, data_dict):
            self.data = data_dict
        
        def __len__(self):
            return len(self.data["observations"])
        
        def __getitem__(self, idx):
            return {
                key: tensor[idx] for key, tensor in self.data.items()
            }
    
    dataset = TransitionDataset(dataset)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def create_sequence_dataloader(
    data: Dict[str, List[np.ndarray]],
    batch_size: int = 32,
    shuffle: bool = True,
    device: torch.device = None,
) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoader for sequence data
    
    Args:
        data: Dictionary containing sequence data
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        device: Device to move tensors to
    
    Returns:
        PyTorch DataLoader
    """
    device = device or get_device()
    
    # Convert to tensors
    dataset = {
        "observations": torch.FloatTensor(np.array(data["observations"])),
        "actions": torch.FloatTensor(np.array(data["actions"])),
        "rewards": torch.FloatTensor(np.array(data["rewards"])),
    }
    
    # Create dataset class
    class SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, data_dict):
            self.data = data_dict
        
        def __len__(self):
            return len(self.data["observations"])
        
        def __getitem__(self, idx):
            return {
                key: tensor[idx] for key, tensor in self.data.items()
            }
    
    dataset = SequenceDataset(dataset)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def normalize_data(data: Dict[str, List[np.ndarray]]) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Any]]:
    """
    Normalize data using mean and std
    
    Args:
        data: Dictionary containing data arrays
    
    Returns:
        Tuple of (normalized_data, normalization_stats)
    """
    normalized_data = {}
    normalization_stats = {}
    
        for key, values in data.items():
        if key in ["observations", "actions", "next_observations"]:
            values_array = np.array(values)
            mean = np.mean(values_array, axis=0)
            std = np.std(values_array, axis=0) + 1e-8  # Add small epsilon
            
            normalized_values = (values_array - mean) / std
            normalized_data[key] = normalized_values.tolist()
            
            normalization_stats[key] = {
                "mean": mean,
                "std": std,
            }
        else:
            normalized_data[key] = values
    
    return normalized_data, normalization_stats


def denormalize_data(data: np.ndarray, stats: Dict[str, Any], key: str) -> np.ndarray:
    """
    Denormalize data using stored statistics
    
    Args:
        data: Normalized data
        stats: Normalization statistics
        key: Key for the data type
        
    Returns:
        Denormalized data
    """
    if key in stats:
        mean = stats[key]["mean"]
        std = stats[key]["std"]
        return data * std + mean
    else:
        return data