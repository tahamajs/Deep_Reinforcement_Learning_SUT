import torch
import numpy as np
from collections import deque
from typing import Dict, Union, List

class ReplayBuffer:
    """
    A simple replay buffer to store environment transitions.
    """
    def __init__(self, capacity: int, observation_shape: tuple, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.device = device

        self.observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32) # Using float for consistency with torch

        self.ptr = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_observation: np.ndarray # Not explicitly stored, but needed for transition
    ):
        # Store current observation, action, reward, done. Next observation is implicitly handled
        # when sampling sequences.
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _sample_indices(self, batch_size: int, sequence_length: int) -> np.ndarray:
        """
        Samples starting indices for sequences from the buffer.
        Ensures that a full sequence can be extracted from the buffer.
        """
        if self.size < sequence_length:
            raise ValueError("Replay buffer does not contain enough data for a sequence.")

        # Exclude the last (sequence_length - 1) elements because a full sequence cannot be formed.
        # Also exclude elements close to the buffer wrap-around point if the sequence would cross it.
        high = self.size - sequence_length + 1
        if high <= 0:
             raise ValueError(f"Not enough samples for sequence length {sequence_length}. Current size: {self.size}")

        indices = np.random.randint(0, high, batch_size)

        # Filter out sequences that wrap around the buffer boundary
        valid_indices = []
        for idx in indices:
            if idx + sequence_length <= self.size or \
               (idx + sequence_length > self.capacity and idx < self.ptr and idx + sequence_length - self.capacity < self.ptr):
                valid_indices.append(idx)
            else:
                # Attempt to find another valid index if the current one is invalid
                # This is a simple retry. More robust solutions might resample or pre-filter more carefully.
                new_idx_attempts = 0
                while new_idx_attempts < 100: # Limit attempts to prevent infinite loop
                    new_idx = np.random.randint(0, high, 1)[0]
                    if new_idx + sequence_length <= self.size or \
                       (new_idx + sequence_length > self.capacity and new_idx < self.ptr and new_idx + sequence_length - self.capacity < self.ptr):
                        valid_indices.append(new_idx)
                        break
                    new_idx_attempts += 1
                if new_idx_attempts == 100:
                    print("Warning: Could not find enough valid sequence indices without wrap-around.")

        return np.array(valid_indices)


    def sample(self, batch_size: int, sequence_length: int) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of sequences from the replay buffer.

        Args:
            batch_size (int): Number of sequences to sample.
            sequence_length (int): Length of each sequence.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing batched sequences of observations, actions, rewards, and dones.
        """
        indices = self._sample_indices(batch_size, sequence_length)

        sampled_obs = []
        sampled_actions = []
        sampled_rewards = []
        sampled_dones = []

        for idx in indices:
            # Handle wrap-around for sequence sampling
            end_idx = idx + sequence_length
            if end_idx <= self.size:
                # Continuous block in buffer
                obs_seq = self.observations[idx:end_idx]
                action_seq = self.actions[idx:end_idx]
                reward_seq = self.rewards[idx:end_idx]
                done_seq = self.dones[idx:end_idx]
            else:
                # Sequence wraps around (e.g., from end to beginning of buffer)
                # This path should ideally be avoided by _sample_indices or handled more carefully
                # For now, we'll assume _sample_indices ensures contiguous valid chunks or wraps correctly.
                # A more robust solution for wrap-around would be to concatenate two slices.
                # For simplicity, assuming _sample_indices guarantees non-wrapping contiguous blocks for now.
                # This is a simplification; a production-grade buffer needs careful wrap-around logic.
                first_part_len = self.capacity - idx
                second_part_len = sequence_length - first_part_len

                obs_seq = np.concatenate((self.observations[idx:self.capacity], self.observations[0:second_part_len]), axis=0)
                action_seq = np.concatenate((self.actions[idx:self.capacity], self.actions[0:second_part_len]), axis=0)
                reward_seq = np.concatenate((self.rewards[idx:self.capacity], self.rewards[0:second_part_len]), axis=0)
                done_seq = np.concatenate((self.dones[idx:self.capacity], self.dones[0:second_part_len]), axis=0)

            sampled_obs.append(obs_seq)
            sampled_actions.append(action_seq)
            sampled_rewards.append(reward_seq)
            sampled_dones.append(done_seq)

        # Convert to torch tensors and move to device
        batch = {
            'observations': torch.as_tensor(np.array(sampled_obs), dtype=torch.float32).to(self.device),
            'actions': torch.as_tensor(np.array(sampled_actions), dtype=torch.float32).to(self.device),
            'rewards': torch.as_tensor(np.array(sampled_rewards), dtype=torch.float32).to(self.device),
            'dones': torch.as_tensor(np.array(sampled_dones), dtype=torch.float32).to(self.device),
        }

        return batch


class DreamerFuNDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset wrapper for the ReplayBuffer, allowing batching and sequence sampling.
    """
    def __init__(self, replay_buffer: ReplayBuffer, sequence_length: int):
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length

    def __len__(self):
        # The effective length of the dataset for sampling full sequences
        if self.replay_buffer.size < self.sequence_length:
            return 0
        return self.replay_buffer.size - self.sequence_length + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single sequence starting from `idx`.
        Note: This is for single item retrieval, `sample` method of ReplayBuffer is for batching.
        """
        if idx + self.sequence_length > self.replay_buffer.size:
            raise IndexError("Index out of bounds for sequence length.")

        # This logic is simplified; for a circular buffer, it would need to handle wrap-around.
        obs_seq = self.replay_buffer.observations[idx : idx + self.sequence_length]
        action_seq = self.replay_buffer.actions[idx : idx + self.sequence_length]
        reward_seq = self.replay_buffer.rewards[idx : idx + self.sequence_length]
        done_seq = self.replay_buffer.dones[idx : idx + self.sequence_length]

        sample = {
            'observations': torch.as_tensor(obs_seq, dtype=torch.float32),
            'actions': torch.as_tensor(action_seq, dtype=torch.float32),
            'rewards': torch.as_tensor(reward_seq, dtype=torch.float32),
            'dones': torch.as_tensor(done_seq, dtype=torch.float32),
        }
        return sample

def create_dataloader(
    replay_buffer: ReplayBuffer,
    sequence_length: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    Creates a PyTorch DataLoader for the Dreamer-FuN ReplayBuffer.

    Args:
        replay_buffer (ReplayBuffer): The replay buffer instance.
        sequence_length (int): Length of sequences to be sampled.
        batch_size (int): Number of sequences in each batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        torch.utils.data.DataLoader: Configured PyTorch DataLoader.
    """
    dataset = DreamerFuNDataset(replay_buffer, sequence_length)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if replay_buffer.device.type == 'cuda' else False,
        drop_last=True
    )
    return dataloader


