"""
Sequence Environment for Testing Temporal World Models

This module implements a sequence prediction environment for testing
recurrent state space models and temporal world modeling capabilities.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Any, Dict, List
import torch


class SequenceEnvironment(gym.Env):
    """Sequence prediction environment for testing temporal world models"""

    def __init__(self, memory_size: int = 5, sequence_length: int = 20):
        super().__init__()
        self.memory_size = memory_size
        self.sequence_length = sequence_length
        self.current_step = 0
        
        # Action space: discrete actions for sequence generation
        self.action_space = spaces.Discrete(4)
        
        # Observation space: current sequence state
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(memory_size,), dtype=np.float32
        )
        
        # Internal state
        self.sequence = []
        self.target_sequence = []
        self.memory = np.zeros(memory_size, dtype=np.float32)
        
        self.name = "SequenceEnvironment"

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Generate target sequence
        self.target_sequence = self._generate_target_sequence()
        
        # Initialize current sequence
        self.sequence = []
        self.memory = np.zeros(self.memory_size, dtype=np.float32)
        self.current_step = 0
        
        return self.memory.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take one step in the environment"""
        # Add action to sequence
        self.sequence.append(action)
        
        # Update memory (sliding window)
        if len(self.sequence) <= self.memory_size:
            self.memory[:len(self.sequence)] = self.sequence
        else:
            self.memory = np.array(self.sequence[-self.memory_size:], dtype=np.float32)
        
        # Check termination
        self.current_step += 1
        terminated = len(self.sequence) >= self.sequence_length
        truncated = False
        
        # Compute reward
        reward = self._compute_reward()
        
        return self.memory.copy(), reward, terminated, truncated, {}

    def _generate_target_sequence(self) -> List[int]:
        """Generate a target sequence for the episode"""
        # Simple pattern: alternating sequence with some complexity
        pattern_length = min(8, self.sequence_length // 2)
        base_pattern = [0, 1, 2, 3] * (pattern_length // 4)
        if pattern_length % 4 != 0:
            base_pattern.extend([0, 1, 2, 3][:pattern_length % 4])
        
        # Repeat pattern to fill sequence
        target = []
        while len(target) < self.sequence_length:
            target.extend(base_pattern)
        
        return target[:self.sequence_length]

    def _compute_reward(self) -> float:
        """Compute reward based on sequence matching"""
        if len(self.sequence) == 0:
            return 0.0
        
        # Reward for matching target sequence
        correct_matches = 0
        for i, (pred, target) in enumerate(zip(self.sequence, self.target_sequence)):
            if pred == target:
                correct_matches += 1
        
        # Normalize by sequence length
        accuracy = correct_matches / len(self.sequence)
        
        # Bonus for completing the sequence
        completion_bonus = 1.0 if len(self.sequence) == self.sequence_length else 0.0
        
        return accuracy + completion_bonus

    def render(self, mode: str = 'human') -> Any:
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Sequence: {self.sequence}")
            print(f"Target: {self.target_sequence[:len(self.sequence)]}")
            print(f"Memory: {self.memory}")
        return None

    def close(self):
        """Close the environment"""
        pass


class SequenceEnvironmentWrapper(gym.Wrapper):
    """Wrapper for SequenceEnvironment with additional features"""

    def __init__(self, env: SequenceEnvironment, normalize_obs: bool = True):
        super().__init__(env)
        self.normalize_obs = normalize_obs

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset with optional normalization"""
        obs, info = self.env.reset(**kwargs)
        if self.normalize_obs:
            # Normalize to [-1, 1] range
            obs = (obs - 1.5) / 1.5
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with optional normalization"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.normalize_obs:
            # Normalize to [-1, 1] range
            obs = (obs - 1.5) / 1.5
        return obs, reward, terminated, truncated, info


class MultiSequenceEnvironment(gym.Env):
    """Multi-sequence environment for testing complex temporal patterns"""

    def __init__(self, num_sequences: int = 3, sequence_length: int = 15):
        super().__init__()
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.current_step = 0
        
        # Action space: discrete actions
        self.action_space = spaces.Discrete(4)
        
        # Observation space: current state of all sequences
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_sequences,), dtype=np.float32
        )
        
        # Internal state
        self.sequences = [[] for _ in range(num_sequences)]
        self.target_sequences = []
        self.current_sequence_idx = 0
        
        self.name = "MultiSequenceEnvironment"

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Generate target sequences
        self.target_sequences = [self._generate_target_sequence() for _ in range(self.num_sequences)]
        
        # Initialize sequences
        self.sequences = [[] for _ in range(self.num_sequences)]
        self.current_sequence_idx = 0
        self.current_step = 0
        
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take one step in the environment"""
        # Add action to current sequence
        self.sequences[self.current_sequence_idx].append(action)
        
        # Switch to next sequence if current one is complete
        if len(self.sequences[self.current_sequence_idx]) >= self.sequence_length:
            self.current_sequence_idx = (self.current_sequence_idx + 1) % self.num_sequences
        
        # Check termination
        self.current_step += 1
        all_complete = all(len(seq) >= self.sequence_length for seq in self.sequences)
        terminated = all_complete
        truncated = self.current_step >= self.sequence_length * self.num_sequences
        
        # Compute reward
        reward = self._compute_reward()
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        """Get observation from current state"""
        obs = np.zeros(self.num_sequences, dtype=np.float32)
        for i, seq in enumerate(self.sequences):
            if len(seq) > 0:
                obs[i] = seq[-1] / 3.0  # Normalize to [0, 1]
        else:
                obs[i] = 0.0
        return obs

    def _generate_target_sequence(self) -> List[int]:
        """Generate a target sequence"""
        # Different patterns for different sequences
        patterns = [
            [0, 1, 2, 3] * 4,  # Simple repetition
            [0, 1, 0, 2, 0, 3] * 3,  # Alternating pattern
            [0, 0, 1, 1, 2, 2, 3, 3] * 2  # Doubled pattern
        ]
        
        pattern = patterns[self.current_sequence_idx % len(patterns)]
        target = []
        while len(target) < self.sequence_length:
            target.extend(pattern)
        
        return target[:self.sequence_length]

    def _compute_reward(self) -> float:
        """Compute reward based on sequence matching"""
        total_reward = 0.0
        
        for i, (seq, target) in enumerate(zip(self.sequences, self.target_sequences)):
            if len(seq) == 0:
                continue
            
            # Count correct matches
            correct_matches = 0
            for j, (pred, targ) in enumerate(zip(seq, target)):
                if pred == targ:
                    correct_matches += 1
            
            # Normalize by sequence length
            accuracy = correct_matches / len(seq)
            total_reward += accuracy
        
        # Average reward across sequences
        return total_reward / self.num_sequences

    def render(self, mode: str = 'human') -> Any:
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Current sequence: {self.current_sequence_idx}")
            for i, (seq, target) in enumerate(zip(self.sequences, self.target_sequences)):
                print(f"Sequence {i}: {seq}")
                print(f"Target {i}: {target[:len(seq)]}")
        return None

    def close(self):
        """Close the environment"""
        pass