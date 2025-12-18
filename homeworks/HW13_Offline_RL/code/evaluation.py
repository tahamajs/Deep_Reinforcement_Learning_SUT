"""
Evaluation utilities for Offline RL experiments.
Includes simple offline dataset loader and a policy evaluation stub.
"""

from typing import Tuple
import numpy as np
import torch


def make_offline_dataset(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
):
    """Create simple dict-based offline dataset used by notebooks/tests."""
    return {
        "states": torch.from_numpy(states).float(),
        "actions": torch.from_numpy(actions).float(),
        "rewards": torch.from_numpy(rewards).float(),
        "next_states": torch.from_numpy(next_states).float(),
        "dones": torch.from_numpy(dones).float(),
    }


def evaluate_policy(
    env, policy_fn, episodes: int = 10, max_steps: int = 1000
) -> Tuple[float, float]:
    """Run policy in gym-like environment to get mean and std of episode returns.

    policy_fn: callable(state) -> action (numpy)
    """
    returns = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done and steps < max_steps:
            a = policy_fn(s)
            s, r, done, _ = env.step(a)
            total += r
            steps += 1
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


if __name__ == "__main__":
    print("evaluation utilities: make_offline_dataset, evaluate_policy")

