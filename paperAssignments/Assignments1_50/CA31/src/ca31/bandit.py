"""Simple multi-armed Bernoulli bandit and a basic Epsilon-Greedy agent.

This module is intentionally small and dependency-free (numpy-only) so it is
simple to test and import-safe in notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, List

import numpy as np


@dataclass
class BernoulliBandit:
    """n-armed Bernoulli bandit.

    Args:
        probs: sequence of success probabilities for each arm.
    """

    probs: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.probs) == 0:
            raise ValueError("Bandit must have at least one arm")
        for p in self.probs:
            if not (0.0 <= p <= 1.0):
                raise ValueError("Each arm probability must be in [0, 1]")

    def __repr__(self) -> str:  # helpful for debugging and clean tests
        return f"BernoulliBandit(probs={list(self.probs)})"

    @property
    def n_arms(self) -> int:
        return len(self.probs)

    def pull(self, arm: int, rng: np.random.Generator) -> int:
        """Pull a single arm and return reward 0 or 1."""
        if not (0 <= arm < self.n_arms):
            raise IndexError("Arm index out of bounds")
        return int(rng.random() < float(self.probs[arm]))


class EpsilonGreedyAgent:
    """Simple epsilon-greedy learner with incremental average updates."""

    def __init__(self, n_arms: int, epsilon: float = 0.1) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        self.n_arms = n_arms
        self.epsilon = float(epsilon)
        self.counts: List[int] = [0] * n_arms
        self.values: List[float] = [0.0] * n_arms

    def select_action(self, rng: np.random.Generator) -> int:
        """Select an action according to epsilon-greedy policy.

        Uses the provided numpy Generator for all randomness so callers can
        control determinism by passing a seeded `Generator`.
        """
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.n_arms))
        # argmax tie-breaking: choose smallest index among ties
        return int(np.argmax(self.values))

    def update(self, action: int, reward: float) -> None:
        """Incrementally update the estimated value for the chosen action."""
        if not (0 <= action < self.n_arms):
            raise IndexError("Action out of bounds")
        self.counts[action] += 1
        n = self.counts[action]
        # incremental mean
        self.values[action] += (float(reward) - self.values[action]) / n

    def best_action(self) -> int:
        """Return current best action by estimated value (ties -> lowest idx)."""
        return int(np.argmax(self.values))
