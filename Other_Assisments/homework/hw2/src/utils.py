"""
Policy Gradient Utilities

This module contains utility functions for processing trajectories and paths.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import numpy as np


def pathlength(path):
    """Get the length of a trajectory path.

    Args:
        path: trajectory path dictionary

    Returns:
        int: length of the path
    """
    return len(path["reward"])


def flatten_list_of_rollouts(paths):
    """Flatten a list of trajectory paths into arrays.

    Args:
        paths: list of trajectory path dictionaries

    Returns:
        tuple: (observations, actions, rewards)
            - observations: array of shape (total_timesteps, ob_dim)
            - actions: array of shape (total_timesteps, ac_dim)
            - rewards: array of shape (total_timesteps,)
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    rewards = np.concatenate([path["reward"] for path in paths])

    return observations, actions, rewards


def add_baseline_to_path(path, baseline):
    """Add baseline predictions to a path.

    Args:
        path: trajectory path dictionary
        baseline: baseline predictions array

    Returns:
        dict: path with baseline added
    """
    path_copy = path.copy()
    path_copy["baseline"] = baseline
    return path_copy


def add_discounted_returns_to_path(path, gamma):
    """Add discounted returns to a path.

    Args:
        path: trajectory path dictionary
        gamma: discount factor

    Returns:
        dict: path with discounted returns added
    """
    path_copy = path.copy()
    rewards = path["reward"]
    returns = []

    for t in range(len(rewards)):
        discounted_return = 0
        for t_prime in range(t, len(rewards)):
            discounted_return += (gamma ** (t_prime - t)) * rewards[t_prime]
        returns.append(discounted_return)

    path_copy["returns"] = np.array(returns)
    return path_copy


def add_value_predictions_to_path(path, values):
    """Add value predictions to a path.

    Args:
        path: trajectory path dictionary
        values: value predictions array

    Returns:
        dict: path with value predictions added
    """
    path_copy = path.copy()
    path_copy["values"] = values
    return path_copy


def add_advantages_to_path(path, gamma, lam=1.0):
    """Add advantages to a path using GAE.

    Args:
        path: trajectory path dictionary
        gamma: discount factor
        lam: GAE lambda parameter

    Returns:
        dict: path with advantages added
    """
    path_copy = path.copy()
    rewards = path["reward"]
    values = path["values"]

    # Compute TD residuals
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    deltas = np.append(deltas, rewards[-1] - values[-1])  # Last step

    # Compute advantages using GAE
    advantages = []
    advantage = 0
    for delta in reversed(deltas):
        advantage = delta + gamma * lam * advantage
        advantages.append(advantage)
    advantages.reverse()

    path_copy["advantages"] = np.array(advantages)
    return path_copy


def compute_advantages(re_n, gamma, reward_to_go=True, normalize_advantages=True):
    """Compute advantages for a batch of trajectories.

    Args:
        re_n: list of reward arrays
        gamma: discount factor
        reward_to_go: whether to use reward-to-go
        normalize_advantages: whether to normalize advantages

    Returns:
        adv_n: advantages array
    """
    adv_n = []

    for re in re_n:
        if reward_to_go:
            # Reward-to-go advantages
            adv_path = []
            for t in range(len(re)):
                adv_t = 0
                for t_prime in range(t, len(re)):
                    adv_t += (gamma ** (t_prime - t)) * re[t_prime]
                adv_path.append(adv_t)
            adv_n.extend(adv_path)
        else:
            # Trajectory-based advantages
            total_return = sum(
                gamma**t_prime * re[t_prime] for t_prime in range(len(re))
            )
            adv_n.extend([total_return] * len(re))

    adv_n = np.array(adv_n)

    # Normalize advantages
    if normalize_advantages:
        adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)

    return adv_n


def compute_qvals(re_n, gamma, reward_to_go=True):
    """Compute Q-values for a batch of trajectories.

    Args:
        re_n: list of reward arrays
        gamma: discount factor
        reward_to_go: whether to use reward-to-go

    Returns:
        q_n: Q-values array
    """
    q_n = []

    for re in re_n:
        if reward_to_go:
            # Reward-to-go Q-values
            q_path = []
            for t in range(len(re)):
                q_t = 0
                for t_prime in range(t, len(re)):
                    q_t += (gamma ** (t_prime - t)) * re[t_prime]
                q_path.append(q_t)
            q_n.extend(q_path)
        else:
            # Trajectory-based Q-values
            total_return = sum(
                gamma**t_prime * re[t_prime] for t_prime in range(len(re))
            )
            q_n.extend([total_return] * len(re))

    return np.array(q_n)
