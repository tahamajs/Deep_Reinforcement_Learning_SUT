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


def collect_sequence_data(
    env,
    episodes: int = 50,
    episode_length: int = 20,
    seed: int = 42,
    random_action_prob: float = 0.3,
    action_noise: float = 0.05,
) -> Dict[str, List[np.ndarray]]:
    """Collect sequential data from the environment for RSSM/Dreamer training.

    The function generates fixed-length sequences of observations, actions, rewards,
    and done flags by interacting with the environment using a stochastic policy.

    Args:
        env: Gymnasium-compatible environment.
        episodes: Number of rollouts to collect.
        episode_length: Maximum length of each collected sequence.
        seed: Random seed used for reproducibility.
        random_action_prob: Probability of taking a completely random action.
        action_noise: Standard deviation of Gaussian noise added to previous action
            for continuous control tasks.

    Returns:
        Dictionary containing lists of numpy arrays with shape
        (episode_length, *obs_dim) for observations and similarly for other keys.
    """

    set_seed(seed)

    obs_shape = env.observation_space.shape
    action_shape = getattr(env.action_space, "shape", ())
    is_discrete_action = len(action_shape) == 0

    zero_obs = np.zeros(obs_shape, dtype=np.float32)
    zero_action = 0 if is_discrete_action else np.zeros(action_shape, dtype=np.float32)

    sequences = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_obs: List[np.ndarray] = []
        episode_actions: List[np.ndarray] = []
        episode_rewards: List[float] = []
        episode_dones: List[bool] = []

        prev_action: Optional[np.ndarray] = None

        for step in range(episode_length):
            take_random = random.random() < random_action_prob or prev_action is None

            if take_random:
                action = env.action_space.sample()
            else:
                if is_discrete_action:
                    action = prev_action
                else:
                    noise = np.random.normal(0, action_noise, size=action_shape)
                    action = np.clip(
                        prev_action + noise,
                        env.action_space.low,
                        env.action_space.high,
                    )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            episode_obs.append(np.array(obs, dtype=np.float32))
            if is_discrete_action:
                episode_actions.append(np.array(action, dtype=np.int64))
            else:
                episode_actions.append(np.array(action, dtype=np.float32))
            episode_rewards.append(float(reward))
            episode_dones.append(done)

            obs = next_obs
            prev_action = (
                np.array(action, copy=True)
                if not is_discrete_action
                else np.array(action, dtype=np.int64)
            )

            if done:
                prev_action = None
                break

        if len(episode_obs) == 0:
            continue

        def _pad_sequence(values: List[Any], target_len: int, pad_value: Any):
            if len(values) >= target_len:
                return values[:target_len]
            padded = list(values)
            while len(padded) < target_len:
                if len(padded) == 0:
                    padded.append(np.array(pad_value, copy=True))
                else:
                    padded.append(np.array(padded[-1], copy=True))
            return padded

        padded_obs = _pad_sequence(episode_obs, episode_length, zero_obs)
        padded_actions = _pad_sequence(episode_actions, episode_length, zero_action)
        padded_rewards = _pad_sequence(
            [np.array(r, dtype=np.float32) for r in episode_rewards],
            episode_length,
            np.array(0.0, dtype=np.float32),
        )
        padded_dones = _pad_sequence(
            [np.array(d, dtype=bool) for d in episode_dones],
            episode_length,
            np.array(True, dtype=bool),
        )

        sequences["observations"].append(np.stack(padded_obs))
        sequences["actions"].append(np.stack(padded_actions))
        sequences["rewards"].append(np.stack(padded_rewards))
        sequences["dones"].append(np.stack(padded_dones))

    return sequences


def create_data_loader(*args, **kwargs):
    """Alias for :func:`create_dataloader` for backwards compatibility."""

    return create_dataloader(*args, **kwargs)


def split_data(
    data: Dict[str, List[Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Split data dictionary into train/validation/test partitions."""

    if not data:
        return {}, {}, {}

    first_key = next(iter(data))
    total_samples = len(data[first_key])
    if total_samples == 0:
        return {k: [] for k in data}, {k: [] for k in data}, {k: [] for k in data}

    indices = list(range(total_samples))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def _subset(idxs: List[int]) -> Dict[str, List[Any]]:
        return {key: [data[key][i] for i in idxs] for key in data}

    # Ensure split dictionaries are explicit copies to avoid accidental mutations
    train_data = _subset(train_idx)
    val_data = _subset(val_idx)
    test_data = _subset(test_idx)

    return train_data, val_data, test_data


def augment_data(
    data: Dict[str, List[Any]],
    noise_std: float = 0.01,
    include_original: bool = True,
    seed: int = 42,
) -> Dict[str, List[Any]]:
    """Apply lightweight data augmentation to numeric entries in the dataset."""

    if not data:
        return {}

    rng = np.random.default_rng(seed)
    augmented = {key: [] for key in data}

    total_samples = len(next(iter(data.values())))

    for idx in range(total_samples):
        for key, values in data.items():
            value = values[idx]
            array_value = np.array(value)
            if array_value.dtype.kind in {"f", "i", "u"}:
                noise = rng.normal(0.0, noise_std, size=array_value.shape)
                augmented_value = array_value + noise
                augmented[key].append(augmented_value.astype(array_value.dtype, copy=False))
            else:
                augmented[key].append(array_value)

    if include_original:
        combined = {key: [] for key in data}
        for key, values in data.items():
            combined[key].extend([np.array(v) for v in values])
            combined[key].extend(augmented[key])
        return combined

    return augmented


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