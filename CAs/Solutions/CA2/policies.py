"""
Policy Classes for Reinforcement Learning

This module contains various policy implementations for the GridWorld environment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

np.random.seed(42)
random.seed(42)


class Policy:
    """
    Base class for policies in reinforcement learning.

    A policy defines how an agent selects actions given states.
    """

    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        """Get action for a given state"""
        raise NotImplementedError("Subclasses must implement get_action")

    def get_action_probabilities(self, state):
        """Get probability distribution over actions for a state"""
        raise NotImplementedError("Subclasses must implement get_action_probabilities")


class RandomPolicy(Policy):
    """
    A policy that selects actions uniformly at random.
    """

    def get_action(self, state):
        """Select a random valid action"""
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return None
        return np.random.choice(valid_actions)

    def get_action_probabilities(self, state):
        """Return uniform probabilities over valid actions"""
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return {}
        prob = 1.0 / len(valid_actions)
        return {action: prob for action in valid_actions}


class GreedyPolicy(Policy):
    """
    A greedy policy that always selects the action with highest Q-value.
    """

    def __init__(self, env, q_values=None):
        super().__init__(env)
        self.q_values = q_values or defaultdict(float)

    def get_action(self, state):
        """Select action with highest Q-value"""
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return None

        best_action = None
        best_value = float("-inf")

        for action in valid_actions:
            q_val = self.q_values[(state, action)]
            if q_val > best_value:
                best_value = q_val
                best_action = action

        return best_action

    def get_action_probabilities(self, state):
        """Return probability 1 for best action, 0 for others"""
        best_action = self.get_action(state)
        valid_actions = self.env.get_valid_actions(state)

        probs = {}
        for action in valid_actions:
            probs[action] = 1.0 if action == best_action else 0.0

        return probs


class CustomPolicy(Policy):
    """
    A custom policy that implements a specific strategy.
    This policy prefers moving right and down when possible.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_preference = ["right", "down", "left", "up"]

    def get_action(self, state):
        """Select action based on custom preference order"""
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return None

        for action in self.action_preference:
            if action in valid_actions:
                return action

        return np.random.choice(valid_actions)

    def get_action_probabilities(self, state):
        """Return probability 1 for preferred action, 0 for others"""
        preferred_action = self.get_action(state)
        valid_actions = self.env.get_valid_actions(state)

        probs = {}
        for action in valid_actions:
            probs[action] = 1.0 if action == preferred_action else 0.0

        return probs


class GreedyActionPolicy(Policy):
    """
    A policy that selects actions greedily based on a value function.
    This policy considers the immediate reward and next state value.
    """

    def __init__(self, env, values=None, gamma=0.9):
        super().__init__(env)
        self.values = values or defaultdict(float)
        self.gamma = gamma

    def get_action(self, state):
        """Select action that maximizes Q(s,a) = R + gamma * V(s')"""
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return None

        best_action = None
        best_value = float("-inf")

        for action in valid_actions:

            transitions = self.env.P[state][action]
            expected_value = 0

            for prob, next_state, reward in transitions:
                expected_value += prob * (reward + self.gamma * self.values[next_state])

            if expected_value > best_value:
                best_value = expected_value
                best_action = action

        return best_action

    def get_action_probabilities(self, state):
        """Return probability 1 for best action, 0 for others"""
        best_action = self.get_action(state)
        valid_actions = self.env.get_valid_actions(state)

        probs = {}
        for action in valid_actions:
            probs[action] = 1.0 if action == best_action else 0.0

        return probs


def create_policy(policy_type, env, **kwargs):
    """Factory function to create policies"""
    if policy_type == "random":
        return RandomPolicy(env)
    elif policy_type == "greedy":
        return GreedyPolicy(env, **kwargs)
    elif policy_type == "custom":
        return CustomPolicy(env)
    elif policy_type == "greedy_action":
        return GreedyActionPolicy(env, **kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
