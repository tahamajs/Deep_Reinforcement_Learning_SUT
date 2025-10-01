"""
Custom Environments for DQN Testing and Analysis
===============================================

This module provides custom environments designed specifically for
testing and analyzing DQN behavior, including environments for:
- Overestimation bias demonstration
- Sample efficiency testing
- State space complexity analysis
- Exploration challenges

Author: CA5 Implementation
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import random
from typing import Tuple, Dict, Any


class OverestimationTestEnv(gym.Env):
    """
    Custom environment to demonstrate overestimation bias in Q-learning.
    
    This environment is designed to create situations where Q-learning
    naturally overestimates values, allowing comparison between
    standard DQN and Double DQN.
    """
    
    def __init__(self, stochastic_reward=True, noise_level=0.1):
        super().__init__()
        
        self.stochastic_reward = stochastic_reward
        self.noise_level = noise_level
        
        # Simple 1D state space with 10 states
        self.n_states = 10
        self.n_actions = 3
        
        self.observation_space = spaces.Box(
            low=0, high=self.n_states-1, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Terminal state and max steps
        self.terminal_state = self.n_states - 1
        self.max_steps = 50
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.state = 0
        self.steps = 0
        
        return np.array([self.state], dtype=np.float32), {}
    
    def step(self, action):
        """Take action and return next state, reward, done, info"""
        self.steps += 1
        
        # Transition function with some randomness
        if action == 0:  # Move left
            self.state = max(0, self.state - 1)
        elif action == 1:  # Stay
            pass  # No movement
        elif action == 2:  # Move right
            self.state = min(self.n_states - 1, self.state + 1)
        
        # Add some random noise to transitions
        if random.random() < 0.1:
            self.state = random.randint(0, self.n_states - 1)
        
        # Reward function designed to create overestimation
        reward = self._get_reward(action)
        
        # Terminal conditions
        done = (self.state == self.terminal_state) or (self.steps >= self.max_steps)
        
        return np.array([self.state], dtype=np.float32), reward, done, False, {}
    
    def _get_reward(self, action):
        """Reward function that creates overestimation bias"""
        # Base reward depends on state and action
        base_reward = self.state * 0.1
        
        # Action-specific bonuses that create bias
        action_bonus = {
            0: -0.1,  # Left movement penalty
            1: 0.0,   # Neutral
            2: 0.1    # Right movement bonus
        }[action]
        
        reward = base_reward + action_bonus
        
        # Add stochastic noise if enabled
        if self.stochastic_reward:
            noise = np.random.normal(0, self.noise_level)
            reward += noise
        
        # Terminal state bonus
        if self.state == self.terminal_state:
            reward += 1.0
        
        return reward


class SampleEfficiencyTestEnv(gym.Env):
    """
    Environment designed to test sample efficiency of different DQN variants.
    
    Features sparse rewards and requires learning long-term dependencies
    to evaluate how well different prioritization schemes work.
    """
    
    def __init__(self, grid_size=5, sparse_reward=True):
        super().__init__()
        
        self.grid_size = grid_size
        self.sparse_reward = sparse_reward
        
        # 2D grid world
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # Up, Right, Down, Left
        
        # Goal position (top-right corner)
        self.goal_pos = (grid_size-1, grid_size-1)
        self.max_steps = grid_size * grid_size * 2
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset to random starting position"""
        super().reset(seed=seed)
        
        # Start at bottom-left corner
        self.pos = (0, 0)
        self.steps = 0
        
        return np.array(self.pos, dtype=np.float32), {}
    
    def step(self, action):
        """Take action in grid world"""
        self.steps += 1
        
        # Movement actions: 0=Up, 1=Right, 2=Down, 3=Left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dy, dx = moves[action]
        
        new_y = np.clip(self.pos[0] + dy, 0, self.grid_size - 1)
        new_x = np.clip(self.pos[1] + dx, 0, self.grid_size - 1)
        
        self.pos = (new_y, new_x)
        
        # Reward calculation
        reward = self._get_reward()
        
        # Terminal conditions
        done = (self.pos == self.goal_pos) or (self.steps >= self.max_steps)
        
        return np.array(self.pos, dtype=np.float32), reward, done, False, {}
    
    def _get_reward(self):
        """Calculate reward based on position"""
        if self.sparse_reward:
            # Sparse reward: only at goal
            return 10.0 if self.pos == self.goal_pos else -0.01
        else:
            # Dense reward: distance to goal
            dist = abs(self.pos[0] - self.goal_pos[0]) + abs(self.pos[1] - self.goal_pos[1])
            return -dist * 0.1


class ExplorationTestEnv(gym.Env):
    """
    Environment to test exploration capabilities of different DQN variants.
    
    Features multiple local optima and requires good exploration to find
    the global optimum.
    """
    
    def __init__(self, n_arms=10, optimal_arm=7):
        super().__init__()
        
        self.n_arms = n_arms
        self.optimal_arm = optimal_arm
        
        # Single state, multiple actions (multi-armed bandit variant)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_arms)
        
        # Reward distributions for each arm
        self.arm_means = np.random.normal(0, 0.5, n_arms)
        self.arm_means[optimal_arm] = 1.0  # Optimal arm
        
        # Create some local optima
        local_optima = [1, 3, 8]
        for arm in local_optima:
            if arm != optimal_arm and arm < n_arms:
                self.arm_means[arm] = 0.7
        
        self.max_steps = 1000
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.steps = 0
        return np.array([0.5], dtype=np.float32), {}
    
    def step(self, action):
        """Pull arm and get reward"""
        self.steps += 1
        
        # Get reward from selected arm
        reward = np.random.normal(self.arm_means[action], 0.1)
        
        # Episode ends after max steps
        done = self.steps >= self.max_steps
        
        return np.array([0.5], dtype=np.float32), reward, done, False, {}


def create_test_environment(env_type="cartpole"):
    """
    Create test environment for DQN training.
    
    Args:
        env_type: Type of environment ("cartpole", "overestimation", 
                 "sample_efficiency", "exploration")
    
    Returns:
        env: Gymnasium environment
        state_size: Size of state space
        action_size: Size of action space
    """
    if env_type == "cartpole":
        env = gym.make("CartPole-v1")
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
    elif env_type == "overestimation":
        env = OverestimationTestEnv()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
    elif env_type == "sample_efficiency":
        env = SampleEfficiencyTestEnv()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
    elif env_type == "exploration":
        env = ExplorationTestEnv()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    return env, state_size, action_size


def create_atari_environment(game_name="PongNoFrameskip-v4"):
    """
    Create Atari environment with standard preprocessing.
    
    Args:
        game_name: Name of Atari game
    
    Returns:
        env: Preprocessed Atari environment
        state_size: State dimensions after preprocessing
        action_size: Number of available actions
    """
    try:
        import ale_py
        gym.register_envs(ale_py)
    except ImportError:
        print("Warning: ale-py not installed. Atari environments not available.")
        return create_test_environment("cartpole")
    
    env = gym.make(game_name)
    
    # Apply standard Atari preprocessing
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )
    
    env = gym.wrappers.FrameStack(env, 4)
    
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    
    return env, state_size, action_size


class EnvironmentWrapper:
    """
    Wrapper class to provide additional functionality for DQN training.
    """
    
    def __init__(self, env):
        self.env = env
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def reset(self):
        """Reset environment and tracking"""
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
        
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        return self.env.reset()
    
    def step(self, action):
        """Step environment and update tracking"""
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if done or truncated:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
        
        return obs, reward, done, truncated, info
    
    def get_statistics(self):
        """Get environment statistics"""
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "total_episodes": len(self.episode_rewards)
        }
    
    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped environment"""
        return getattr(self.env, name)