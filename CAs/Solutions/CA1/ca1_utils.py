import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Any


def set_seed(seed: int = 42) -> None:
    """Set seeds for numpy, torch and python random for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def moving_average(x: List[float], window: int = 10) -> np.ndarray:
    """Compute moving average using a convolution. Returns same-length array."""
    if len(x) < 1:
        return np.array([])
    if window <= 1:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def gym_reset(env) -> np.ndarray:
    """Handle different gym reset return signatures (state or (state, info))."""
    result = env.reset()
    if isinstance(result, tuple):
        state, _ = result
    else:
        state = result
    return np.array(state, dtype=np.float32)


def gym_step(env, action: Any) -> Tuple[np.ndarray, float, bool, dict]:
    """Handle different gym step return signatures and agent action types.

    Returns: next_state, reward, done, info
    """

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
