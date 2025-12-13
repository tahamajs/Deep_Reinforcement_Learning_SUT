import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.config import EnvironmentConfig
from collections import deque
from typing import Union

class DreamerFuNEnvWrapper(gym.Wrapper):
    """
    A wrapper for Gymnasium environments to standardize observations and actions
    for the Dreamer-FuN agent.
    Handles image preprocessing (resizing, grayscale, normalization) and action space flattening.
    """
    def __init__(self, env: gym.Env, config: EnvironmentConfig):
        super().__init__(env)
        self.config = config
        self.original_env = env # Keep a reference to the original env

        # Standardize observation space
        self.observation_space = self._get_observation_space(env.observation_space)

        # Standardize action space
        self.action_space = self._get_action_space(env.action_space)

        self.reward_range = (env.reward_range[0] * config.reward_scale, env.reward_range[1] * config.reward_scale)
        self.episode_buffer = deque(maxlen=config.time_limit) # For sequence storage if needed

    def _get_observation_space(self, obs_space: gym.Space) -> gym.Space:
        """
        Transforms the observation space to a consistent format.
        """
        if isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 3: # Image observations
            # Assuming image observations are (H, W, C) from Gymnasium, converting to (C, H, W)
            # and potentially resizing/grayscaling
            c, h, w = obs_space.shape[2], self.config.image_size[0], self.config.image_size[1]
            if self.config.grayscale: # Convert to grayscale if specified
                c = 1
            return spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)
        elif isinstance(obs_space, spaces.Box): # Vector observations
            return spaces.Box(low=obs_space.low, high=obs_space.high, shape=obs_space.shape, dtype=np.float32)
        else:
            raise NotImplementedError(f"Observation space {obs_space} not supported yet.")

    def _get_action_space(self, action_space: gym.Space) -> gym.Space:
        """
        Transforms the action space to a consistent format (e.g., continuous for continuous, flattened for discrete).
        """
        if isinstance(action_space, spaces.Discrete):
            # For discrete actions, we might output a one-hot vector or just the scalar action ID
            # For Dreamer, actions are often continuous or one-hot for discrete
            return spaces.Box(low=0, high=action_space.n - 1, shape=(1,), dtype=np.int64) # Return scalar action ID
        elif isinstance(action_space, spaces.Box): # Continuous actions
            return spaces.Box(low=action_space.low, high=action_space.high, shape=action_space.shape, dtype=np.float32)
        else:
            raise NotImplementedError(f"Action space {action_space} not supported yet.")

    def _preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Applies preprocessing steps to the observation.
        """
        if isinstance(self.original_env.observation_space, spaces.Box) and len(self.original_env.observation_space.shape) == 3: # Image observations
            from PIL import Image
            # Assuming obs is (H, W, C)
            img = Image.fromarray(obs)
            if self.config.image_size != (obs.shape[0], obs.shape[1]):
                img = img.resize(self.config.image_size, Image.ANTIALIAS)
            if self.config.grayscale:
                img = img.convert('L')
            
            obs = np.array(img).transpose(2, 0, 1) if not self.config.grayscale else np.array(img).unsqueeze(0)
            obs = obs.astype(np.float32) / 255.0 # Normalize to [0, 1]
            return obs
        elif isinstance(self.original_env.observation_space, spaces.Box): # Vector observations
            return obs.astype(np.float32)
        return obs

    def step(self, action: Union[int, np.ndarray]):
        """
        Steps the environment with the given action.
        """
        if isinstance(self.original_env.action_space, spaces.Discrete) and isinstance(action, np.ndarray):
            # Convert numpy array action (e.g., [0]) to scalar int (e.g., 0)
            action = int(action.item())
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._preprocess_observation(obs)
        scaled_reward = reward * self.config.reward_scale
        done = terminated or truncated
        return processed_obs, scaled_reward, done, info

    def reset(self, **kwargs):
        """
        Resets the environment.
        """
        obs, info = self.env.reset(**kwargs)
        processed_obs = self._preprocess_observation(obs)
        return processed_obs, info

def make_env(env_name: str, config: EnvironmentConfig, seed: int = None) -> gym.Env:
    """
    Creates and wraps a Gymnasium environment.
    """
    env = gym.make(env_name)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    wrapped_env = DreamerFuNEnvWrapper(env, config)
    return wrapped_env


