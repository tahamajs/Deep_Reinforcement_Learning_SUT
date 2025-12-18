"""
Environment utilities for CA8.
Contains ToTheMaxWrapper that applies the "To the Max" reward transform.
"""

from typing import Optional, Tuple, Any

import gymnasium as gym


class ToTheMaxWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that computes a simple progress-based bonus and
    returns the transformed reward r' = max(r, beta * 1[progress > 0]).

    Exposes raw reward under info['reward_raw'] and transformed under info['reward_max'].
    Works with envs that include a 'dist_to_goal' key in the info dict on reset/step.
    """

    def __init__(self, env: gym.Env, beta: float = 0.4):
        super().__init__(env)
        self.beta = beta
        self.prev_dist: Optional[float] = None

    def reset(self, **kwargs) -> Tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        # Some envs return (obs, info); others return obs only. Normalize.
        if isinstance(info, dict):
            self.prev_dist = info.get("dist_to_goal", None)
        else:
            try:
                self.prev_dist = info.get("dist_to_goal", None)
            except Exception:
                self.prev_dist = None
        return obs, info

    def step(self, action):
        res = self.env.step(action)
        # Gymnasium: obs, reward, terminated, truncated, info
        if len(res) == 5:
            obs, r, terminated, truncated, info = res
            done = terminated or truncated
        else:
            # Backwards compatibility: support older gym tuple
            obs, r, done, info = res

        dist = info.get("dist_to_goal", None)
        progress = 0.0
        if dist is not None and self.prev_dist is not None:
            progress = float(self.prev_dist - dist)
        bonus = float(self.beta) if progress > 0 else 0.0
        r_max = max(r, bonus)
        # update prev_dist for next step
        self.prev_dist = dist
        info = dict(info)  # copy to avoid mutating env internals
        info["reward_raw"] = r
        info["reward_max"] = r_max

        if len(res) == 5:
            return obs, r_max, terminated, truncated, info
        else:
            return obs, r_max, done, info












