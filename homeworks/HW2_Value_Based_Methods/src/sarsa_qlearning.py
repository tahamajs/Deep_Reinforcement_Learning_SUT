"""
SARSA and Q-Learning implementations for discrete environments.

This module provides complete, import-safe implementations of:
- initialize_q
- greedy and epsilon-greedy action selection
- q_learning
- sarsa

Designed for OpenAI Gym/Gymnasium discrete observation/action spaces.
Type-hinted and dependency-light (numpy).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def initialize_q(env) -> np.ndarray:
    """
    Initialize a Q-table for a discrete environment.
    Raises ValueError if observation or action space is not discrete.
    """
    try:
        n_states = env.observation_space.n
        n_actions = env.action_space.n
    except Exception as e:
        raise ValueError(
            "Environment must have discrete observation and action spaces"
        ) from e
    return np.zeros((n_states, n_actions), dtype=float)


def greedy_action(q: np.ndarray, state: int) -> int:
    """Return greedy action for given state using Q-table."""
    return int(np.argmax(q[state]))


def epsilon_greedy_action(
    q: np.ndarray, state: int, epsilon: float, rng: Optional[np.random.Generator] = None
) -> int:
    """Epsilon-greedy action selection."""
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < epsilon:
        return int(rng.integers(0, q.shape[1]))
    return greedy_action(q, state)


def q_learning(
    env,
    episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tabular Q-learning for discrete environments.
    Returns (Q-table, episode_rewards_array).
    """
    rng = np.random.default_rng(seed)
    q = initialize_q(env)
    rewards = np.zeros(episodes, dtype=float)

    for ep in range(episodes):
        obs, info = env.reset(seed=(None if seed is None else int(seed + ep)))
        state = int(obs)
        done = False
        eps = max(epsilon_end, epsilon_start * (epsilon_decay**ep))
        total_r = 0.0
        while not done:
            action = epsilon_greedy_action(q, state, eps, rng)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            next_state = int(next_obs)
            done = terminated or truncated
            # Q-learning update
            td_target = reward + gamma * np.max(q[next_state]) * (0.0 if done else 1.0)
            td_error = td_target - q[state, action]
            q[state, action] += alpha * td_error
            state = next_state
            total_r += reward
        rewards[ep] = total_r
    return q, rewards


def sarsa(
    env,
    episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tabular SARSA algorithm (on-policy).
    Returns (Q-table, episode_rewards_array).
    """
    rng = np.random.default_rng(seed)
    q = initialize_q(env)
    rewards = np.zeros(episodes, dtype=float)

    for ep in range(episodes):
        obs, info = env.reset(seed=(None if seed is None else int(seed + ep)))
        state = int(obs)
        done = False
        eps = max(epsilon_end, epsilon_start * (epsilon_decay**ep))
        action = epsilon_greedy_action(q, state, eps, rng)
        total_r = 0.0
        while not done:
            next_obs, reward, terminated, truncated, _info = env.step(action)
            next_state = int(next_obs)
            done = terminated or truncated
            next_action = (
                epsilon_greedy_action(q, next_state, eps, rng) if not done else 0
            )
            td_target = reward + gamma * q[next_state, next_action] * (
                0.0 if done else 1.0
            )
            td_error = td_target - q[state, action]
            q[state, action] += alpha * td_error
            state = next_state
            action = next_action
            total_r += reward
        rewards[ep] = total_r
    return q, rewards


if __name__ == "__main__":
    # Lightweight self-check (no heavy compute). Only run when invoked directly.
    import gymnasium as gym

    env = gym.make("Taxi-v3")
    q_q, r_q = q_learning(env, episodes=10, seed=0)
    q_s, r_s = sarsa(env, episodes=10, seed=0)
    print("Q-learning sample rewards:", r_q[:3])
    print("SARSA sample rewards:", r_s[:3])
