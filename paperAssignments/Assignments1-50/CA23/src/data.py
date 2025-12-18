"""Minimal data collection helpers for on-policy rollouts.

To keep imports lightweight, ``gym`` is imported lazily inside functions so the
module remains import-safe if ``gym`` is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any, Sequence

import numpy as np
import torch


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    done: bool
    next_obs: np.ndarray


def _unpack_reset(reset_out: Any):
    # Newer gym versions return (obs, info)
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0]
    return reset_out


def _unpack_step(step_out: Any):
    # gym step may return (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
    if len(step_out) == 4:
        obs, reward, done, info = step_out
    elif len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
    else:
        raise RuntimeError("Unrecognized environment.step() return signature")
    return obs, float(reward), bool(done), info


def collect_episodes(env_name: str, policy, num_episodes: int = 1, max_steps: int = 1000) -> List[List[Transition]]:
    """Collect ``num_episodes`` by rolling out ``policy`` in ``env_name``.

    The ``policy`` must implement a ``get_action(obs_tensor)`` method that
    accepts a torch tensor of shape (obs_dim,) or (batch, obs_dim).

    Returns:
        A list of episodes; each episode is a list of :class:`Transition`.
    """
    try:
        import gym
    except Exception as e:  # pragma: no cover - runtime dependency
        raise RuntimeError("`gym` is required to collect rollouts. Install gym and try again.") from e

    env = gym.make(env_name)
    episodes: List[List[Transition]] = []
    for _ in range(num_episodes):
        reset_out = env.reset()
        obs = _unpack_reset(reset_out)
        ep: List[Transition] = []
        for _ in range(max_steps):
            obs_tensor = torch.as_tensor(np.array(obs), dtype=torch.float32)
            action_tensor, _ = policy.get_action(obs_tensor)
            # Support both batched and single returns
            if isinstance(action_tensor, torch.Tensor):
                action = int(action_tensor.squeeze().item())
            else:
                action = int(action_tensor)
            step_out = env.step(action)
            next_obs, reward, done, _info = _unpack_step(step_out)
            ep.append(Transition(obs=np.array(obs), action=action, reward=float(reward), done=done, next_obs=np.array(next_obs)))
            obs = next_obs
            if done:
                break
        episodes.append(ep)
    env.close()
    return episodes


def discounts(rewards: Sequence[float], gamma: float) -> List[float]:
    """Compute discounted returns for a single episode.

    This implementation is simple and clear; for larger workloads consider a
    vectorized implementation.
    """
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    R = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    return list(reversed(returns))
