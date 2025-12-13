import numpy as np
import torch
import random
import gymnasium as gym
from collections import deque
from typing import List, Tuple, Any, Optional

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def moving_average(x: List[float], window: int = 10) -> np.ndarray:
    if len(x) < 1:
        return np.array([])
    if window <= 1:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")

def gym_reset(env: gym.Env, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        result = env.reset(seed=seed)
    else:
        result = env.reset()

    if isinstance(result, tuple):
        state, _ = result
    else:
        state = result
    return np.array(state, dtype=np.float32)

def gym_step(env: gym.Env, action: Any) -> Tuple[np.ndarray, float, bool, dict]:
    if isinstance(action, tuple):
        action_to_env = action[0]
    else:
        action_to_env = action

    result = env.step(action_to_env)
    if len(result) == 4:
        next_state, reward, done, info = result
    else:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    return np.array(next_state, dtype=np.float32), float(reward), bool(done), info


