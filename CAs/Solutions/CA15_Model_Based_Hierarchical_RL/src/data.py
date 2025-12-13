import gymnasium as gym
import numpy as np
import random

from typing import Tuple, Any, Optional, Dict
from src.utils import ReplayBuffer, PrioritizedReplayBuffer, to_tensor


class SimpleGridWorld(gym.Env):
    """A simple 2D grid world environment for testing hierarchical RL."""

    def __init__(self, grid_size: int = 10, goal_pos: Tuple[int, int] = (9, 9), sparse_reward: bool = True):
        super().__init__()
        self.grid_size = grid_size
        self.goal_pos = goal_pos
        self.sparse_reward = sparse_reward

        self.observation_space = gym.spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right

        self.agent_pos = np.array([0, 0], dtype=np.int32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        return self.agent_pos, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        prev_pos = self.agent_pos.copy()

        if action == 0:  # Up
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 1:  # Down
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 2:  # Left
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3:  # Right
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        else:
            raise ValueError("Invalid action")

        done = bool(np.array_equal(self.agent_pos, self.goal_pos))
        reward = 0.0
        if done:
            reward = 1.0  # Goal reached
        elif not self.sparse_reward:
            # Dense reward: negative distance to goal
            reward = -np.linalg.norm(self.agent_pos - self.goal_pos) / (self.grid_size * 2)

        # Optional: small negative reward for each step to encourage efficiency
        reward -= 0.01

        return self.agent_pos, reward, done, False, {}

    def render(self, mode: str = 'human'):
        if mode == 'human':
            grid = np.zeros((self.grid_size, self.grid_size))
            grid[self.goal_pos[0], self.goal_pos[1]] = 2  # Goal
            grid[self.agent_pos[0], self.agent_pos[1]] = 1  # Agent
            print(grid)
        else:
            super().render(mode=mode)

    def close(self):
        pass

