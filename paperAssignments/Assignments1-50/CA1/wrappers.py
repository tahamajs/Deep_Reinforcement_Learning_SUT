"""Common Atari / pixel preprocessing wrappers (lightweight).

This file provides a small set of wrappers suitable for Gym/Gymnasium Atari environments.
It uses the active gym module from `env_utils` to remain compatible with both libraries.
"""

from collections import deque
import cv2
import numpy as np

from paperAssignments.Assignments1_50.CA1.env_utils import gym


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, info = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        # take max over last two frames (for flickering)
        max_frame = (
            np.maximum(self._obs_buffer[-1], self._obs_buffer[-2])
            if len(self._obs_buffer) >= 2
            else self._obs_buffer[-1]
        )
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        self._obs_buffer.append(obs)
        return obs


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        obs_space = env.observation_space
        if len(obs_space.shape) == 3:
            c = 1 if grayscale else obs_space.shape[2]
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.height, self.width, c), dtype=np.uint8
            )

    def observation(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        if self.grayscale and obs.ndim == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            obs = np.expand_dims(obs, -1)
        return obs


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)


def make_atari_env(
    env_id: str, noop_max: int = 30, skip: int = 4, frame_stack: int = 4
):
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=skip)
    env = WarpFrame(env, width=84, height=84, grayscale=True)
    env = ClipRewardEnv(env)
    # frame stack using Gym's FrameStack wrapper
    env = gym.wrappers.FrameStack(env, num_stack=frame_stack)
    return env











