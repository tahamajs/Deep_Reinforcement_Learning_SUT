"""
Continuous CartPole Environment

This module implements a continuous version of the CartPole environment
for testing world models and model-based RL algorithms.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Any, Dict
import torch


class ContinuousCartPole(gym.Env):
    """Continuous CartPole environment for world model testing"""

    def __init__(self, max_steps: int = 500):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: continuous force applied to cart
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        # Physics parameters
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.pole_length = 0.5
        self.pole_mass_length = self.pole_mass * self.pole_length
        self.force_magnitude = 10.0
        self.tau = 0.02  # Time step
        
        # State bounds
        self.x_threshold = 2.4
        self.theta_threshold = 12 * 2 * np.pi / 360
        
        self.name = "ContinuousCartPole"

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize state
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.current_step = 0
        
        return self.state.astype(np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take one step in the environment"""
        # Extract action
        force = action[0] * self.force_magnitude
        
        # Get current state
        x, x_dot, theta, theta_dot = self.state
        
        # Compute derivatives
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        # Physics calculations
        temp = (force + self.pole_mass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.pole_length * (4.0/3.0 - self.pole_mass * costheta**2 / self.total_mass)
        )
        xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass
        
        # Update state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        # Check termination conditions
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )
        
        # Check truncation
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # Compute reward
        reward = self._compute_reward()
        
        return self.state.astype(np.float32), reward, terminated, truncated, {}

    def _compute_reward(self) -> float:
        """Compute reward based on current state"""
        x, x_dot, theta, theta_dot = self.state
        
        # Reward for keeping pole upright and cart centered
        reward = 1.0
        
        # Penalty for large angles
        angle_penalty = abs(theta) / self.theta_threshold
        reward -= angle_penalty * 0.1
        
        # Penalty for large positions
        position_penalty = abs(x) / self.x_threshold
        reward -= position_penalty * 0.1
        
        # Penalty for large velocities
        velocity_penalty = (abs(x_dot) + abs(theta_dot)) * 0.01
        reward -= velocity_penalty
        
        return reward

    def render(self, mode: str = 'human') -> Any:
        """Render the environment (placeholder)"""
        if mode == 'human':
            print(f"State: {self.state}")
        return None

    def close(self):
        """Close the environment"""
        pass


class ContinuousCartPoleWrapper(gym.Wrapper):
    """Wrapper for ContinuousCartPole with additional features"""

    def __init__(self, env: ContinuousCartPole, normalize_obs: bool = True):
        super().__init__(env)
        self.normalize_obs = normalize_obs
        
        if normalize_obs:
            # Normalize observations
            self.obs_mean = np.array([0.0, 0.0, 0.0, 0.0])
            self.obs_std = np.array([2.4, 3.0, 0.2, 3.0])
            self.obs_std = np.where(self.obs_std == 0, 1.0, self.obs_std)

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