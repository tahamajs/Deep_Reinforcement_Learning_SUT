"""
Sequence Environment for RSSM Testing
"""

import numpy as np
from collections import deque


class SequenceEnvironment:
    """Environment that requires memory (partial observability)"""

    def __init__(self, obs_dim=4, memory_length=5):
        self.obs_dim = obs_dim
        self.memory_length = memory_length
        self.action_dim = 2  # Left or right

        self.state = None
        self.memory = None
        self.step_count = 0
        self.max_steps = 50

        self.reset()

    def reset(self):
        """Reset environment"""
        self.state = np.zeros(self.obs_dim)
        self.memory = deque(maxlen=self.memory_length)
        self.step_count = 0

        # Initialize with random values
        for _ in range(self.memory_length):
            self.memory.append(np.random.rand())

        return self._get_observation()

    def _get_observation(self):
        """Get partial observation (doesn't include full memory)"""
        # Only return current state + partial memory information
        recent_memory = list(self.memory)[-2:]  # Only last 2 memory items

        obs = np.concatenate([
            self.state,
            recent_memory + [0.0] * (2 - len(recent_memory))
        ])

        return obs[:self.obs_dim]

    def step(self, action):
        """Take environment step"""
        if isinstance(action, np.ndarray) and action.ndim > 0:
            action = action.item()

        # Discrete action: 0 = left, 1 = right
        action = int(action > 0.5) if isinstance(action, float) else int(action)

        # Update memory based on action
        if action == 0:  # Left
            new_memory_val = max(0.0, list(self.memory)[-1] - 0.1)
        else:  # Right
            new_memory_val = min(1.0, list(self.memory)[-1] + 0.1)

        self.memory.append(new_memory_val)

        # Update state (simple dynamics)
        self.state[0] = new_memory_val
        self.state[1] = np.mean(list(self.memory))
        self.state[2] = action
        self.state[3] = self.step_count / self.max_steps

        # Reward based on memory sequence
        memory_sequence = list(self.memory)
        if len(memory_sequence) >= 3:
            # Reward for maintaining values in middle range
            recent_avg = np.mean(memory_sequence[-3:])
            reward = 1.0 - abs(recent_avg - 0.5) * 2  # Max reward when avg = 0.5
        else:
            reward = 0.0

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_observation(), reward, done