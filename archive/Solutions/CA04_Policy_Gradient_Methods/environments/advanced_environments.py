"""
Advanced Environment Wrappers and Complex Environments
CA4: Policy Gradient Methods and Neural Networks in RL - Advanced Implementation
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any, Dict, List, Optional, Callable
import cv2
from collections import deque
import random
import math


class AtariWrapper(gym.Wrapper):
    """Atari Environment Wrapper with preprocessing"""
    
    def __init__(self, env: gym.Env, frame_skip: int = 4, frame_stack: int = 4,
                 resize_shape: Tuple[int, int] = (84, 84), grayscale: bool = True):
        """Initialize Atari wrapper
        
        Args:
            env: Gymnasium environment
            frame_skip: Number of frames to skip
            frame_stack: Number of frames to stack
            resize_shape: Target resize shape
            grayscale: Whether to convert to grayscale
        """
        super(AtariWrapper, self).__init__(env)
        
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.resize_shape = resize_shape
        self.grayscale = grayscale
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=frame_stack)
        
        # Update observation space
        if grayscale:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(frame_stack, *resize_shape), dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(frame_stack, 3, *resize_shape), dtype=np.uint8
            )
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame
        
        Args:
            frame: Raw frame
            
        Returns:
            Preprocessed frame
        """
        # Convert to grayscale if needed
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize frame
        frame = cv2.resize(frame, self.resize_shape, interpolation=cv2.INTER_AREA)
        
        return frame
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment
        
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        
        # Preprocess initial frame
        processed_frame = self._preprocess_frame(obs)
        
        # Fill frame buffer
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)
        
        return np.array(self.frame_buffer), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        total_reward = 0
        
        # Skip frames
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Preprocess frame
        processed_frame = self._preprocess_frame(obs)
        
        # Update frame buffer
        self.frame_buffer.append(processed_frame)
        
        return np.array(self.frame_buffer), total_reward, terminated, truncated, info


class MultiAgentWrapper(gym.Wrapper):
    """Multi-Agent Environment Wrapper"""
    
    def __init__(self, env: gym.Env, num_agents: int = 2):
        """Initialize multi-agent wrapper
        
        Args:
            env: Gymnasium environment
            num_agents: Number of agents
        """
        super(MultiAgentWrapper, self).__init__(env)
        
        self.num_agents = num_agents
        self.agent_rewards = [0.0] * num_agents
        self.agent_dones = [False] * num_agents
        
        # Update observation and action spaces
        self.observation_space = gym.spaces.Tuple([
            self.env.observation_space for _ in range(num_agents)
        ])
        self.action_space = gym.spaces.Tuple([
            self.env.action_space for _ in range(num_agents)
        ])
    
    def reset(self, **kwargs) -> Tuple[List[np.ndarray], Dict]:
        """Reset environment
        
        Returns:
            Tuple of (observations, info)
        """
        obs, info = self.env.reset(**kwargs)
        
        # Duplicate observation for all agents
        observations = [obs.copy() for _ in range(self.num_agents)]
        
        # Reset agent states
        self.agent_rewards = [0.0] * self.num_agents
        self.agent_dones = [False] * self.num_agents
        
        return observations, info
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], Dict]:
        """Step environment
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # Use first agent's action for environment step
        obs, reward, terminated, truncated, info = self.env.step(actions[0])
        
        # Distribute reward among agents (can be modified for different reward schemes)
        agent_rewards = [reward / self.num_agents] * self.num_agents
        
        # Duplicate observation for all agents
        observations = [obs.copy() for _ in range(self.num_agents)]
        
        # Update agent states
        self.agent_dones = [terminated] * self.num_agents
        self.agent_rewards = agent_rewards
        
        return observations, agent_rewards, self.agent_dones, [truncated] * self.num_agents, info


class CurriculumWrapper(gym.Wrapper):
    """Curriculum Learning Environment Wrapper"""
    
    def __init__(self, env: gym.Env, difficulty_levels: List[Dict] = None,
                 performance_threshold: float = 0.8, window_size: int = 100):
        """Initialize curriculum wrapper
        
        Args:
            env: Gymnasium environment
            difficulty_levels: List of difficulty configurations
            performance_threshold: Performance threshold for level advancement
            window_size: Window size for performance evaluation
        """
        super(CurriculumWrapper, self).__init__(env)
        
        self.difficulty_levels = difficulty_levels or [
            {'name': 'easy', 'max_steps': 200, 'reward_scale': 1.0},
            {'name': 'medium', 'max_steps': 500, 'reward_scale': 1.0},
            {'name': 'hard', 'max_steps': 1000, 'reward_scale': 1.0}
        ]
        
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        
        self.current_level = 0
        self.performance_history = deque(maxlen=window_size)
        self.episode_count = 0
        
        # Apply initial difficulty
        self._apply_difficulty_level()
    
    def _apply_difficulty_level(self):
        """Apply current difficulty level"""
        level = self.difficulty_levels[self.current_level]
        
        # Update max episode steps
        if hasattr(self.env, '_max_episode_steps'):
            self.env._max_episode_steps = level['max_steps']
        
        # Update reward scale
        self.reward_scale = level['reward_scale']
    
    def _update_performance(self, episode_reward: float):
        """Update performance history
        
        Args:
            episode_reward: Episode reward
        """
        self.performance_history.append(episode_reward)
        self.episode_count += 1
        
        # Check if ready to advance level
        if len(self.performance_history) >= self.window_size:
            avg_performance = np.mean(self.performance_history)
            
            if avg_performance >= self.performance_threshold and self.current_level < len(self.difficulty_levels) - 1:
                self.current_level += 1
                self._apply_difficulty_level()
                print(f"Advanced to difficulty level: {self.difficulty_levels[self.current_level]['name']}")
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment
        
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        info['difficulty_level'] = self.current_level
        info['level_name'] = self.difficulty_levels[self.current_level]['name']
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Scale reward
        reward *= self.reward_scale
        
        # Update performance if episode ended
        if terminated or truncated:
            self._update_performance(info.get('episode_reward', 0))
        
        info['difficulty_level'] = self.current_level
        info['level_name'] = self.difficulty_levels[self.current_level]['name']
        
        return obs, reward, terminated, truncated, info


class NoisyObservationWrapper(gym.Wrapper):
    """Add noise to observations for robustness testing"""
    
    def __init__(self, env: gym.Env, noise_std: float = 0.1, noise_type: str = 'gaussian'):
        """Initialize noisy observation wrapper
        
        Args:
            env: Gymnasium environment
            noise_std: Noise standard deviation
            noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper')
        """
        super(NoisyObservationWrapper, self).__init__(env)
        
        self.noise_std = noise_std
        self.noise_type = noise_type
    
    def _add_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add noise to observation
        
        Args:
            obs: Observation
            
        Returns:
            Noisy observation
        """
        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_std, obs.shape)
            return np.clip(obs + noise, self.observation_space.low, self.observation_space.high)
        
        elif self.noise_type == 'uniform':
            noise = np.random.uniform(-self.noise_std, self.noise_std, obs.shape)
            return np.clip(obs + noise, self.observation_space.low, self.observation_space.high)
        
        elif self.noise_type == 'salt_pepper':
            noisy_obs = obs.copy()
            salt_pepper = np.random.random(obs.shape)
            noisy_obs[salt_pepper < self.noise_std/2] = self.observation_space.low
            noisy_obs[salt_pepper > 1 - self.noise_std/2] = self.observation_space.high
            return noisy_obs
        
        else:
            return obs
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment
        
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        noisy_obs = self._add_noise(obs)
        return noisy_obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        noisy_obs = self._add_noise(obs)
        return noisy_obs, reward, terminated, truncated, info


class RewardShapingWrapper(gym.Wrapper):
    """Reward shaping wrapper for better learning"""
    
    def __init__(self, env: gym.Env, shaping_function: Callable = None,
                 shaping_weight: float = 1.0):
        """Initialize reward shaping wrapper
        
        Args:
            env: Gymnasium environment
            shaping_function: Function to compute shaped reward
            shaping_weight: Weight for shaped reward
        """
        super(RewardShapingWrapper, self).__init__(env)
        
        self.shaping_function = shaping_function
        self.shaping_weight = shaping_weight
        self.previous_state = None
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment
        
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        self.previous_state = obs
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add shaped reward
        if self.shaping_function is not None and self.previous_state is not None:
            shaped_reward = self.shaping_function(self.previous_state, action, obs, reward)
            reward += self.shaping_weight * shaped_reward
        
        self.previous_state = obs
        return obs, reward, terminated, truncated, info


class CustomMountainCarEnv(gym.Env):
    """Custom Mountain Car Environment with additional features"""
    
    def __init__(self, goal_velocity: float = 0.0, power: float = 0.0015,
                 max_steps: int = 200, render_mode: str = None):
        """Initialize custom mountain car environment
        
        Args:
            goal_velocity: Goal velocity threshold
            power: Power of the car
            max_steps: Maximum steps per episode
            render_mode: Render mode
        """
        super(CustomMountainCarEnv, self).__init__()
        
        self.goal_velocity = goal_velocity
        self.power = power
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # State bounds
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        
        # Action space
        self.action_space = gym.spaces.Discrete(3)  # 0: left, 1: nothing, 2: right
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=np.array([self.min_position, -self.max_speed]),
            high=np.array([self.max_position, self.max_speed]),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.step_count = 0
        
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Random initial position
        self.state = np.array([
            self.np_random.uniform(low=-0.6, high=-0.4),
            0.0
        ], dtype=np.float32)
        
        self.step_count = 0
        
        return self.state, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        position, velocity = self.state
        
        # Apply action
        force = (action - 1) * self.power
        
        # Update velocity
        velocity += force * math.cos(3 * position)
        velocity *= 0.9995  # Friction
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        
        # Update position
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        
        # Reset velocity if hit left wall
        if position == self.min_position and velocity < 0:
            velocity = 0
        
        self.state = np.array([position, velocity], dtype=np.float32)
        self.step_count += 1
        
        # Check termination conditions
        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        truncated = bool(self.step_count >= self.max_steps)
        
        # Compute reward
        if terminated:
            reward = 100.0
        else:
            # Shaped reward based on position and velocity
            reward = -1.0 + 0.1 * position + 0.1 * abs(velocity)
        
        info = {
            'position': position,
            'velocity': velocity,
            'step_count': self.step_count
        }
        
        return self.state, reward, terminated, truncated, info
    
    def render(self):
        """Render environment"""
        if self.render_mode == 'human':
            print(f"Position: {self.state[0]:.3f}, Velocity: {self.state[1]:.3f}")


class CustomPendulumEnv(gym.Env):
    """Custom Pendulum Environment with continuous actions"""
    
    def __init__(self, max_steps: int = 200, render_mode: str = None):
        """Initialize custom pendulum environment
        
        Args:
            max_steps: Maximum steps per episode
            render_mode: Render mode
        """
        super(CustomPendulumEnv, self).__init__()
        
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Physical parameters
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.dt = 0.05
        
        # Action space (continuous torque)
        self.action_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space (cos(theta), sin(theta), angular velocity)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -8.0]),
            high=np.array([1.0, 1.0, 8.0]),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.step_count = 0
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Random initial angle
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.state[1] = self.np_random.uniform(low=-1, high=1)
        
        self.step_count = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        """Get observation
        
        Returns:
            Observation array
        """
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        theta, thetadot = self.state
        u = action[0]
        
        # Apply torque
        u = np.clip(u, -2.0, 2.0)
        
        # Physics update
        thetadotdot = -(3 * self.g / (2 * self.l)) * np.sin(theta) + (3.0 / (self.m * self.l**2)) * u
        thetadot = thetadot + thetadotdot * self.dt
        theta = theta + thetadot * self.dt
        
        self.state = np.array([theta, thetadot])
        self.step_count += 1
        
        # Check termination
        terminated = False
        truncated = bool(self.step_count >= self.max_steps)
        
        # Compute reward
        reward = -(theta**2 + 0.1 * thetadot**2 + 0.001 * u**2)
        
        info = {
            'theta': theta,
            'thetadot': thetadot,
            'step_count': self.step_count
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        """Render environment"""
        if self.render_mode == 'human':
            theta, thetadot = self.state
            print(f"Theta: {theta:.3f}, Angular Velocity: {thetadot:.3f}")


def create_advanced_environment(env_name: str, **kwargs) -> gym.Env:
    """Create advanced environment with wrappers
    
    Args:
        env_name: Environment name
        **kwargs: Additional arguments
        
    Returns:
        Wrapped environment
    """
    # Base environment
    if env_name == 'CustomMountainCar':
        env = CustomMountainCarEnv(**kwargs)
    elif env_name == 'CustomPendulum':
        env = CustomPendulumEnv(**kwargs)
    else:
        env = gym.make(env_name, **kwargs)
    
    # Apply wrappers based on environment type
    if 'Atari' in env_name or 'Breakout' in env_name or 'Pong' in env_name:
        env = AtariWrapper(env, **kwargs.get('atari_kwargs', {}))
    
    if kwargs.get('multi_agent', False):
        env = MultiAgentWrapper(env, kwargs.get('num_agents', 2))
    
    if kwargs.get('curriculum', False):
        env = CurriculumWrapper(env, **kwargs.get('curriculum_kwargs', {}))
    
    if kwargs.get('noisy_obs', False):
        env = NoisyObservationWrapper(env, **kwargs.get('noise_kwargs', {}))
    
    if kwargs.get('reward_shaping', False):
        env = RewardShapingWrapper(env, **kwargs.get('shaping_kwargs', {}))
    
    return env


def get_environment_info(env: gym.Env) -> Dict[str, Any]:
    """Get comprehensive environment information
    
    Args:
        env: Environment
        
    Returns:
        Environment information dictionary
    """
    info = {
        'name': env.spec.id if hasattr(env, 'spec') and env.spec else 'Unknown',
        'observation_space': str(env.observation_space),
        'action_space': str(env.action_space),
        'max_episode_steps': getattr(env, '_max_episode_steps', None),
        'reward_range': getattr(env, 'reward_range', None),
        'metadata': getattr(env, 'metadata', {}),
        'wrappers': []
    }
    
    # Check for wrappers
    current_env = env
    while hasattr(current_env, 'env'):
        wrapper_name = current_env.__class__.__name__
        info['wrappers'].append(wrapper_name)
        current_env = current_env.env
    
    return info

