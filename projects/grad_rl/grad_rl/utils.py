import os
import random
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper, NoopResetEnv, MaxAndSkipEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def set_seed(seed: int | None = None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_atari_env(env_id: str, num_envs: int = 4, seed: int | None = None):
    def make_one(rank: int):
        def _init():
            env = gym.make(env_id, render_mode=None)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = AtariWrapper(env)
            env = Monitor(env)
            if seed is not None:
                env.reset(seed=seed + rank)
            return env

        return _init

    venv = SubprocVecEnv([make_one(i) for i in range(num_envs)])
    venv = VecFrameStack(venv, n_stack=4)
    return venv


def make_env(env_id: str, seed: int | None = None, vec: bool = True, num_envs: int = 1):
    """Create a Gymnasium env; wraps vectorization when requested."""
    def _init(rank: int = 0):
        env = gym.make(env_id)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed + rank)
        return env

    if not vec:
        return _init()

    if num_envs == 1:
        return DummyVecEnv([_init])
    return SubprocVecEnv([lambda r=i: _init(rank=r) for i in range(num_envs)])


def evaluate_sb3(model, env, n_eval_episodes: int = 5, deterministic: bool = True):
    """SB3 helper that returns mean/std reward."""
    mean, std = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=deterministic)
    return {"mean_reward": float(mean), "std_reward": float(std)}
