"""
Environment Wrappers

This module contains common environment wrappers for RL.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import numpy as np
import gym
from collections import deque
class TimeLimitWrapper(gym.Wrapper):
    """Wrapper that limits episode length."""

    def __init__(self, env, max_episode_steps):
        """Initialize time limit wrapper.

        Args:
            env: Environment to wrap
            max_episode_steps: Maximum steps per episode
        """
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0

    def reset(self, **kwargs):
        """Reset environment and step counter."""
        self.elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Take step and check time limit."""
        obs, reward, done, info = self.env.step(action)
        self.elapsed_steps += 1

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True

        return obs, reward, done, info
class ActionRepeatWrapper(gym.Wrapper):
    """Wrapper that repeats actions for multiple steps."""

    def __init__(self, env, repeat=4):
        """Initialize action repeat wrapper.

        Args:
            env: Environment to wrap
            repeat: Number of times to repeat each action
        """
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        """Repeat action for multiple steps."""
        total_reward = 0
        done = False
        info = {}

        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward

            if done:
                break

        return obs, total_reward, done, info
class FrameStackWrapper(gym.Wrapper):
    """Wrapper that stacks multiple frames."""

    def __init__(self, env, num_stack=4):
        """Initialize frame stack wrapper.

        Args:
            env: Environment to wrap
            num_stack: Number of frames to stack
        """
        super().__init__(env)
        self.num_stack = num_stack
        obs_shape = env.observation_space.shape
        self.frames = deque(maxlen=num_stack)

        low = np.repeat(env.observation_space.low, num_stack, axis=-1)
        high = np.repeat(env.observation_space.high, num_stack, axis=-1)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        """Reset environment and frame buffer."""
        obs = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        """Take step and update frame buffer."""
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """Get stacked observation."""
        return np.concatenate(list(self.frames), axis=-1)
class RewardScaleWrapper(gym.RewardWrapper):
    """Wrapper that scales rewards."""

    def __init__(self, env, scale=1.0):
        """Initialize reward scale wrapper.

        Args:
            env: Environment to wrap
            scale: Reward scaling factor
        """
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        """Scale reward."""
        return reward * self.scale
class ObservationWrapper(gym.ObservationWrapper):
    """Base class for observation wrappers."""

    def __init__(self, env):
        """Initialize observation wrapper.

        Args:
            env: Environment to wrap
        """
        super().__init__(env)

    def observation(self, obs):
        """Process observation."""
        return obs
class GrayscaleWrapper(ObservationWrapper):
    """Convert RGB observations to grayscale."""

    def __init__(self, env):
        """Initialize grayscale wrapper."""
        super().__init__(env)

        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(old_shape[0], old_shape[1], 1), dtype=np.uint8
        )

    def observation(self, obs):
        """Convert to grayscale."""

        return np.mean(obs, axis=-1, keepdims=True).astype(np.uint8)
class ResizeWrapper(ObservationWrapper):
    """Resize observations."""

    def __init__(self, env, size=(84, 84)):
        """Initialize resize wrapper.

        Args:
            env: Environment to wrap
            size: Target size (height, width)
        """
        super().__init__(env)
        self.size = size
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=size + (old_shape[-1],),
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs):
        """Resize observation."""
        try:
            from PIL import Image

            img = Image.fromarray(obs)
            img = img.resize(self.size)
            return np.array(img)
        except ImportError:

            h, w = self.size
            return obs.reshape(h, w, -1)