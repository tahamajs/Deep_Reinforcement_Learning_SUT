import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MultivariateNormal, kl_divergence
import torch.multiprocessing as mp
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, deque, namedtuple
import random
import pickle
import json
import copy
import time
import threading
from typing import Tuple, List, Dict, Optional, Union, NamedTuple, Any
import warnings
from dataclasses import dataclass, field
import math
from tqdm import tqdm
from abc import ABC, abstractmethod
import itertools

warnings.filterwarnings("ignore")

# Game Theory Utilities and Basic Multi-Agent Framework


class GameTheoryUtils:
    """Utility class for game-theoretic analysis."""

    @staticmethod
    def find_nash_equilibria(payoff_matrices):
        """
        Find pure strategy Nash equilibria for n-player games.

        Args:
            payoff_matrices: List of payoff matrices, one per player
        Returns:
            List of Nash equilibrium strategy profiles
        """
        n_players = len(payoff_matrices)
        if n_players != 2:
            raise NotImplementedError("Only 2-player games supported")

        matrix_a, matrix_b = payoff_matrices[0], payoff_matrices[1]
        nash_equilibria = []

        rows, cols = matrix_a.shape

        for i in range(rows):
            for j in range(cols):
                # Check if (i,j) is a Nash equilibrium
                is_nash = True

                # Check if player 1 wants to deviate
                for i_prime in range(rows):
                    if matrix_a[i_prime, j] > matrix_a[i, j]:
                        is_nash = False
                        break

                # Check if player 2 wants to deviate
                if is_nash:
                    for j_prime in range(cols):
                        if matrix_b[i, j_prime] > matrix_b[i, j]:
                            is_nash = False
                            break

                if is_nash:
                    nash_equilibria.append((i, j))

        return nash_equilibria

    @staticmethod
    def is_pareto_optimal(payoff_matrices, strategy_profile):
        """Check if a strategy profile is Pareto optimal."""
        current_payoffs = [matrix[strategy_profile] for matrix in payoff_matrices]

        # Check all other strategy profiles
        for profile in itertools.product(
            *[range(matrix.shape[i]) for i, matrix in enumerate(payoff_matrices)]
        ):
            if profile == strategy_profile:
                continue

            candidate_payoffs = [matrix[profile] for matrix in payoff_matrices]

            # Check if candidate dominates current
            dominates = True
            strictly_better = False

            for i in range(len(current_payoffs)):
                if candidate_payoffs[i] < current_payoffs[i]:
                    dominates = False
                    break
                elif candidate_payoffs[i] > current_payoffs[i]:
                    strictly_better = True

            if dominates and strictly_better:
                return False

        return True

    @staticmethod
    def compute_best_response(payoff_matrix, opponent_strategy):
        """Compute best response to opponent's mixed strategy."""
        expected_payoffs = payoff_matrix @ opponent_strategy
        return np.zeros_like(expected_payoffs).at[np.argmax(expected_payoffs)].set(1.0)


class MultiAgentEnvironment:
    """Base class for multi-agent environments."""

    def __init__(self, n_agents, state_dim, action_dim, cooperative=True):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cooperative = cooperative
        self.state = None
        self.step_count = 0
        self.max_steps = 200

    def reset(self):
        """Reset environment to initial state."""
        self.state = np.random.randn(self.state_dim)
        self.step_count = 0
        return [self.state.copy() for _ in range(self.n_agents)]

    def step(self, actions):
        """Execute joint action and return next states, rewards, dones."""
        self.step_count += 1

        # Simple dynamics: state evolves based on joint action
        joint_action = np.mean(actions, axis=0)
        noise = np.random.randn(self.state_dim) * 0.1
        self.state = 0.9 * self.state + 0.1 * joint_action[: self.state_dim] + noise

        # Compute rewards
        if self.cooperative:
            # Cooperative: shared reward based on coordination
            coordination_bonus = -np.mean(
                [
                    np.linalg.norm(actions[i] - joint_action)
                    for i in range(self.n_agents)
                ]
            )
            base_reward = -np.linalg.norm(self.state)  # Drive state to origin
            rewards = [base_reward + coordination_bonus] * self.n_agents
        else:
            # Competitive: individual rewards with competition
            rewards = []
            for i in range(self.n_agents):
                individual_reward = -np.linalg.norm(
                    self.state - actions[i][: self.state_dim]
                )
                competition_penalty = (
                    sum(
                        [
                            np.linalg.norm(actions[i] - actions[j])
                            for j in range(self.n_agents)
                            if j != i
                        ]
                    )
                    * 0.1
                )
                rewards.append(individual_reward - competition_penalty)

        done = self.step_count >= self.max_steps
        next_states = [self.state.copy() for _ in range(self.n_agents)]

        return next_states, rewards, done

    def render(self):
        """Visualize current environment state."""
        pass


# Demonstration of game theory concepts
def demonstrate_game_theory():
    """Demonstrate basic game theory concepts."""
    print("🎯 Game Theory Analysis Demo")

    # Prisoner's Dilemma
    print("\n1. Prisoner's Dilemma:")
    # Player 1's payoff matrix (rows: cooperate, defect)
    prisoner_a = np.array(
        [[-1, -3], [0, -2]]
    )  # (cooperate, defect) vs (cooperate, defect)
    # Player 2's payoff matrix
    prisoner_b = np.array([[-1, 0], [-3, -2]])

    print("Player 1 payoff matrix:")
    print(prisoner_a)
    print("Player 2 payoff matrix:")
    print(prisoner_b)

    nash_eq = GameTheoryUtils.find_nash_equilibria([prisoner_a, prisoner_b])
    print(f"Nash equilibria: {nash_eq}")

    for eq in nash_eq:
        is_pareto = GameTheoryUtils.is_pareto_optimal([prisoner_a, prisoner_b], eq)
        print(f"Strategy {eq}: Pareto optimal = {is_pareto}")

    # Coordination Game
    print("\n2. Coordination Game:")
    coord_a = np.array([[2, 0], [0, 1]])
    coord_b = np.array([[2, 0], [0, 1]])

    print("Coordination game (both players have same payoffs):")
    print(coord_a)

    nash_eq = GameTheoryUtils.find_nash_equilibria([coord_a, coord_b])
    print(f"Nash equilibria: {nash_eq}")

    return prisoner_a, prisoner_b, coord_a, coord_b


# Test multi-agent environment
def test_multi_agent_env():
    """Test the basic multi-agent environment."""
    print("\n🤖 Multi-Agent Environment Test")

    # Cooperative environment
    print("Testing cooperative environment:")
    coop_env = MultiAgentEnvironment(
        n_agents=3, state_dim=4, action_dim=4, cooperative=True
    )
    states = coop_env.reset()
    print(f"Initial states shape: {[s.shape for s in states]}")

    # Random actions
    actions = [np.random.randn(coop_env.action_dim) for _ in range(coop_env.n_agents)]
    next_states, rewards, done = coop_env.step(actions)

    print(f"Rewards (cooperative): {rewards}")
    print(f"All agents get same reward: {len(set(rewards)) == 1}")

    # Competitive environment
    print("\nTesting competitive environment:")
    comp_env = MultiAgentEnvironment(
        n_agents=3, state_dim=4, action_dim=4, cooperative=False
    )
    states = comp_env.reset()
    next_states, rewards, done = comp_env.step(actions)

    print(f"Rewards (competitive): {rewards}")
    print(f"Agents get different rewards: {len(set(rewards)) > 1}")

    return coop_env, comp_env


# Run demonstrations
game_matrices = demonstrate_game_theory()
environments = test_multi_agent_env()

print("\n✅ Game theory and multi-agent foundations implemented!")
