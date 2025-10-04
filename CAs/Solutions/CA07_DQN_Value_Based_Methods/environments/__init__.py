"""
Environment utilities and wrappers for CA07 DQN experiments
===========================================================
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


class RewardShapingWrapper(gym.Wrapper):
    """Wrapper for reward shaping experiments"""

    def __init__(self, env, reward_scale: float = 1.0, reward_shift: float = 0.0):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = reward * self.reward_scale + self.reward_shift
        return obs, shaped_reward, terminated, truncated, info


class StateNormalizationWrapper(gym.Wrapper):
    """Wrapper for state normalization"""

    def __init__(self, env):
        super().__init__(env)
        self.state_mean = None
        self.state_std = None
        self.state_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update_stats(obs)
        return self._normalize_state(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_stats(obs)
        normalized_obs = self._normalize_state(obs)
        return normalized_obs, reward, terminated, truncated, info

    def _update_stats(self, state):
        if self.state_mean is None:
            self.state_mean = np.zeros_like(state)
            self.state_std = np.ones_like(state)

        self.state_count += 1
        alpha = 1.0 / self.state_count
        self.state_mean = (1 - alpha) * self.state_mean + alpha * state
        self.state_std = np.sqrt(
            (1 - alpha) * self.state_std**2 + alpha * (state - self.state_mean) ** 2
        )

    def _normalize_state(self, state):
        if self.state_count < 2:
            return state
        return (state - self.state_mean) / (self.state_std + 1e-8)


class ActionRepeatWrapper(gym.Wrapper):
    """Wrapper for action repetition"""

    def __init__(self, env, repeat: int = 1):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class EpisodeStatisticsWrapper(gym.Wrapper):
    """Wrapper for collecting episode statistics"""

    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def reset(self, **kwargs):
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

        self.current_episode_reward = 0
        self.current_episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_episode_reward += reward
        self.current_episode_length += 1

        if terminated or truncated:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

        return obs, reward, terminated, truncated, info

    def get_statistics(self):
        if not self.episode_rewards:
            return {}

        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "total_episodes": len(self.episode_rewards),
        }


def create_cartpole_env(
    reward_scale: float = 1.0, normalize_states: bool = False
) -> gym.Env:
    """Create CartPole environment with optional wrappers"""
    env = gym.make("CartPole-v1")

    if reward_scale != 1.0:
        env = RewardShapingWrapper(env, reward_scale=reward_scale)

    if normalize_states:
        env = StateNormalizationWrapper(env)

    env = EpisodeStatisticsWrapper(env)
    return env


def create_mountain_car_env(
    reward_scale: float = 1.0, normalize_states: bool = False
) -> gym.Env:
    """Create MountainCar environment with optional wrappers"""
    env = gym.make("MountainCar-v0")

    if reward_scale != 1.0:
        env = RewardShapingWrapper(env, reward_scale=reward_scale)

    if normalize_states:
        env = StateNormalizationWrapper(env)

    env = EpisodeStatisticsWrapper(env)
    return env


def create_acrobot_env(
    reward_scale: float = 1.0, normalize_states: bool = False
) -> gym.Env:
    """Create Acrobot environment with optional wrappers"""
    env = gym.make("Acrobot-v1")

    if reward_scale != 1.0:
        env = RewardShapingWrapper(env, reward_scale=reward_scale)

    if normalize_states:
        env = StateNormalizationWrapper(env)

    env = EpisodeStatisticsWrapper(env)
    return env


def get_environment_info(env_name: str) -> Dict[str, Any]:
    """Get information about an environment"""
    env = gym.make(env_name)

    info = {
        "name": env_name,
        "observation_space": {
            "shape": env.observation_space.shape,
            "dtype": str(env.observation_space.dtype),
            "low": (
                env.observation_space.low.tolist()
                if hasattr(env.observation_space, "low")
                else None
            ),
            "high": (
                env.observation_space.high.tolist()
                if hasattr(env.observation_space, "high")
                else None
            ),
        },
        "action_space": {
            "n": env.action_space.n if hasattr(env.action_space, "n") else None,
            "shape": env.action_space.shape,
            "dtype": str(env.action_space.dtype),
            "low": (
                env.action_space.low.tolist()
                if hasattr(env.action_space, "low")
                else None
            ),
            "high": (
                env.action_space.high.tolist()
                if hasattr(env.action_space, "high")
                else None
            ),
        },
    }

    env.close()
    return info


def test_environment(env_name: str, num_episodes: int = 5) -> Dict[str, Any]:
    """Test an environment by running random episodes"""
    env = gym.make(env_name)

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }

