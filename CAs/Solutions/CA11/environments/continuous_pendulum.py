"""
Continuous Pendulum Environment

This module implements a continuous pendulum environment for testing
world models and model-based RL algorithms.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Any, Dict
import torch


class ContinuousPendulum(gym.Env):
    """Continuous Pendulum environment for world model testing"""

    def __init__(self, max_steps: int = 200):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: continuous torque applied to pendulum
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: [cos(theta), sin(theta), angular_velocity]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Physics parameters
        self.gravity = 9.8
        self.mass = 1.0
        self.length = 1.0
        self.dt = 0.05
        self.max_speed = 8.0
        self.max_torque = 2.0
        
        self.name = "ContinuousPendulum"

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize state (random angle and velocity)
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.current_step = 0
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take one step in the environment"""
        # Extract action
        u = np.clip(action[0], -self.max_torque, self.max_torque)
        
        # Get current state
        th, thdot = self.state
        
        # Physics calculations
        newthdot = thdot + (3 * self.gravity / (2 * self.length) * np.sin(th) + 
                           3.0 / (self.mass * self.length**2) * u) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        
        # Update state
        self.state = np.array([newth, newthdot])
        
        # Check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Compute reward
        reward = self._compute_reward(th, thdot, u)
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        """Get observation from state"""
        th, thdot = self.state
        return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)

    def _compute_reward(self, th: float, thdot: float, u: float) -> float:
        """Compute reward based on current state and action"""
        # Reward for being upright (cos(th) close to 1)
        reward = -np.cos(th)
        
        # Small penalty for angular velocity
        reward -= 0.1 * thdot**2
        
        # Small penalty for action magnitude
        reward -= 0.01 * u**2
        
        return reward

    def render(self, mode: str = 'human') -> Any:
        """Render the environment (placeholder)"""
        if mode == 'human':
            th, thdot = self.state
            print(f"Angle: {th:.3f}, Angular Velocity: {thdot:.3f}")
        return None

    def close(self):
        """Close the environment"""
        pass


class ContinuousPendulumWrapper(gym.Wrapper):
    """Wrapper for ContinuousPendulum with additional features"""

    def __init__(self, env: ContinuousPendulum, normalize_obs: bool = True):
        super().__init__(env)
        self.normalize_obs = normalize_obs
        
        if normalize_obs:
            # Normalize observations
            self.obs_mean = np.array([0.0, 0.0, 0.0])
            self.obs_std = np.array([1.0, 1.0, 8.0])

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset with optional normalization"""
        obs, info = self.env.reset(**kwargs)
        if self.normalize_obs:
            obs = (obs - self.obs_mean) / self.obs_std
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with optional normalization"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.normalize_obs:
            obs = (obs - self.obs_mean) / self.obs_std
        return obs, reward, terminated, truncated, info