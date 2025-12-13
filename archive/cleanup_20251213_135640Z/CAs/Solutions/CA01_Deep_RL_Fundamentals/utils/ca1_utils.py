import json
import os
import platform
import random
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns


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


def _git_commit_hash() -> str:
    """Best-effort retrieval of current git commit hash."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def write_run_info(path: str, hyperparams: Dict[str, Any], extra: Dict[str, Any] | None = None) -> None:
    """Persist run metadata for reproducibility."""
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "commit": _git_commit_hash(),
        "hyperparameters": hyperparams,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    if extra:
        info["extra"] = extra

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
