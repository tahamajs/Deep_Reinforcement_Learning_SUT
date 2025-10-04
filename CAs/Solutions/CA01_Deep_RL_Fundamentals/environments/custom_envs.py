"""
Custom environment implementations for CA1 Deep RL Fundamentals.

This module contains custom environment implementations that can be used
for testing and demonstration purposes.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Any, Dict


class SimpleGridWorld(gym.Env):
    """
    A simple grid world environment for testing RL algorithms.

    The agent starts at (0,0) and needs to reach the goal at (grid_size-1, grid_size-1).
    The agent can move in 4 directions: up, down, left, right.
    """

    def __init__(self, grid_size: int = 5):
        super().__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(2,), dtype=np.int32
        )

        self.agent_pos = None
        self.goal_pos = (grid_size - 1, grid_size - 1)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        return self.agent_pos.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        action_map = {
            0: [-1, 0],  # up
            1: [1, 0],  # down
            2: [0, -1],  # left
            3: [0, 1],  # right
        }

        # Update agent position
        new_pos = self.agent_pos + action_map[action]

        # Check boundaries
        if (new_pos >= 0).all() and (new_pos < self.grid_size).all():
            self.agent_pos = new_pos

        # Calculate reward
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 100.0  # Large reward for reaching goal
            terminated = True
        else:
            reward = -0.1  # Small negative reward for each step
            terminated = False

        truncated = False
        info = {}

        return self.agent_pos.copy(), reward, terminated, truncated, info

    def render(self):
        """Render the current state of the environment."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = "."

        # Mark agent position
        grid[self.agent_pos[0], self.agent_pos[1]] = "A"

        # Mark goal position
        grid[self.goal_pos[0], self.goal_pos[1]] = "G"

        print("\n".join([" ".join(row) for row in grid]))
        print()


class MultiArmedBandit(gym.Env):
    """
    A simple multi-armed bandit environment for testing bandit algorithms.

    The environment has n_arms arms, each with a different reward distribution.
    The goal is to learn which arm gives the highest expected reward.
    """

    def __init__(self, n_arms: int = 10, reward_std: float = 1.0):
        super().__init__()
        self.n_arms = n_arms
        self.action_space = spaces.Discrete(n_arms)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Generate random reward means for each arm
        np.random.seed(42)
        self.true_means = np.random.normal(0, 1, n_arms)
        self.reward_std = reward_std

        self.current_step = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.current_step = 0
        return np.array([0.0]), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Pull the selected arm and receive reward."""
        if action >= self.n_arms:
            raise ValueError(f"Action {action} is invalid. Must be < {self.n_arms}")

        # Sample reward from the selected arm's distribution
        reward = np.random.normal(self.true_means[action], self.reward_std)

        self.current_step += 1

        # Episode ends after a fixed number of steps
        terminated = self.current_step >= 100
        truncated = False

        # Return dummy observation (bandits are stateless)
        obs = np.array([0.0])
        info = {
            "true_mean": self.true_means[action],
            "optimal_arm": np.argmax(self.true_means),
        }

        return obs, reward, terminated, truncated, info

    def get_optimal_arm(self) -> int:
        """Return the index of the optimal arm."""
        return np.argmax(self.true_means)

    def get_arm_means(self) -> np.ndarray:
        """Return the true means of all arms."""
        return self.true_means.copy()


def create_cartpole_env() -> gym.Env:
    """Create and return a CartPole environment."""
    return gym.make("CartPole-v1")


def create_mountain_car_env() -> gym.Env:
    """Create and return a MountainCar environment."""
    return gym.make("MountainCar-v0")


def create_acrobot_env() -> gym.Env:
    """Create and return an Acrobot environment."""
    return gym.make("Acrobot-v1")


def create_simple_grid_world(grid_size: int = 5) -> SimpleGridWorld:
    """Create and return a simple grid world environment."""
    return SimpleGridWorld(grid_size=grid_size)


def create_multi_armed_bandit(n_arms: int = 10) -> MultiArmedBandit:
    """Create and return a multi-armed bandit environment."""
    return MultiArmedBandit(n_arms=n_arms)


