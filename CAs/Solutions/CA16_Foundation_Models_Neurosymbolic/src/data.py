import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict, Any

from src.config import config

class TrajectoryDataset(Dataset):
    """
    A dataset for handling sequences of (state, action, reward, return_to_go, timestep)
    tuples, as required by Decision Transformers.
    """
    def __init__(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
                 returns_to_go: np.ndarray, timesteps: np.ndarray, seq_len: int = config.SEQ_LEN):
        self.states = torch.from_numpy(states).float()
        self.actions = torch.from_numpy(actions).float() # Assuming one-hot or continuous
        self.rewards = torch.from_numpy(rewards).float()
        self.returns_to_go = torch.from_numpy(returns_to_go).float()
        self.timesteps = torch.from_numpy(timesteps).long()
        self.seq_len = seq_len
        
        assert len(self.states) == len(self.actions) == len(self.rewards) == \
               len(self.returns_to_go) == len(self.timesteps), "All data arrays must have same length"

    def __len__(self):
        return len(self.states) - self.seq_len + 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_len
        
        s = self.states[start_idx:end_idx]
        a = self.actions[start_idx:end_idx]
        r = self.rewards[start_idx:end_idx]
        rtg = self.returns_to_go[start_idx:end_idx]
        t = self.timesteps[start_idx:end_idx]
        
        return s, a, r, rtg, t


class MultiModalDataset(Dataset):
    """
    Dataset for environments that provide both visual/continuous observations
    and symbolic predicates.
    """
    def __init__(self, visual_obs: np.ndarray, symbolic_obs: np.ndarray, 
                 actions: np.ndarray, rewards: np.ndarray):
        self.visual_obs = torch.from_numpy(visual_obs).float()
        self.symbolic_obs = torch.from_numpy(symbolic_obs).long() # Assuming discrete/categorical
        self.actions = torch.from_numpy(actions).float()
        self.rewards = torch.from_numpy(rewards).float()

    def __len__(self):
        return len(self.visual_obs)

    def __getitem__(self, idx):
        return self.visual_obs[idx], self.symbolic_obs[idx], self.actions[idx], self.rewards[idx]


def generate_dummy_trajectory_data(
    num_samples: int = 1000,
    state_dim: int = config.STATE_DIM,
    action_dim: int = config.ACTION_DIM,
    seq_len: int = config.SEQ_LEN
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates dummy trajectory data for testing Decision Transformers.
    """
    states = np.random.randn(num_samples, state_dim).astype(np.float32)
    actions = np.random.randint(0, action_dim, size=(num_samples,)).astype(np.int64)
    # Convert actions to one-hot for DT
    actions_one_hot = np.eye(action_dim)[actions].astype(np.float32)
    rewards = np.random.randn(num_samples,).astype(np.float32)
    returns_to_go = np.cumsum(rewards[::-1])[::-1].copy().astype(np.float32) # Calculate returns-to-go
    timesteps = np.arange(num_samples).astype(np.int64)
    
    return states, actions_one_hot, rewards, returns_to_go, timesteps


def generate_dummy_multimodal_data(
    num_samples: int = 1000,
    visual_dim: int = config.STATE_DIM,
    symbolic_dim: int = config.SYMBOLIC_FEATURE_DIM,
    action_dim: int = config.ACTION_DIM
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates dummy multimodal data for testing neurosymbolic agents.
    """
    visual_obs = np.random.randn(num_samples, visual_dim).astype(np.float32)
    symbolic_obs = np.random.randint(0, 2, size=(num_samples, symbolic_dim)).astype(np.int64) # Binary symbolic features
    actions = np.random.randint(0, action_dim, size=(num_samples,)).astype(np.int64)
    # Convert actions to one-hot for consistency if needed, here just keeping as int
    rewards = np.random.randn(num_samples,).astype(np.float32)
    
    return visual_obs, symbolic_obs, actions, rewards

