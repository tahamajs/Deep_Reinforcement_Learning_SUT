"""
GridWorld Environment for Reinforcement Learning

This module contains the GridWorld environment implementation used for
demonstrating MDP concepts and RL algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import random

np.random.seed(42)
random.seed(42)

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12


class GridWorld:
    """
    A simple GridWorld environment for demonstrating MDP concepts.

    The agent starts at (0,0) and tries to reach the goal at (3,3).
    There are obstacles and different reward structures.
    """

    def __init__(self, size=4, goal_reward=10, step_reward=-0.1, obstacle_reward=-5):
        self.size = size
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.obstacle_reward = obstacle_reward

        self.states = [(i, j) for i in range(size) for j in range(size)]
        self.actions = ["up", "down", "left", "right"]
        self.action_effects = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        self.start_state = (0, 0)
        self.goal_state = (3, 3)
        self.obstacles = [(1, 1), (2, 1), (1, 2)]

        self._build_transition_model()

    def _build_transition_model(self):
        """Build transition probability and reward models"""
        self.P = {}

        for state in self.states:
            self.P[state] = {}
            for action in self.actions:
                self.P[state][action] = self._get_transitions(state, action)

    def _get_transitions(self, state, action):
        """Get possible transitions for a state-action pair"""
        if state == self.goal_state:
            return [(1.0, state, 0)]

        if state in self.obstacles:
            return [(1.0, state, self.obstacle_reward)]

        dx, dy = self.action_effects[action]
        next_x, next_y = state[0] + dx, state[1] + dy

        if next_x < 0 or next_x >= self.size or next_y < 0 or next_y >= self.size:
            next_state = state
        else:
            next_state = (next_x, next_y)

        if next_state == self.goal_state:
            reward = self.goal_reward
        elif next_state in self.obstacles:
            reward = self.obstacle_reward
        else:
            reward = self.step_reward

        return [(1.0, next_state, reward)]

    def get_valid_actions(self, state):
        """Get valid actions from a given state"""
        if state == self.goal_state or state in self.obstacles:
            return []
        return self.actions.copy()

    def visualize_grid(self, values=None, policy=None, title="GridWorld"):
        """Visualize the grid world with optional value function or policy"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        grid = np.zeros((self.size, self.size))

        for i, j in self.obstacles:
            grid[i, j] = -1

        goal_i, goal_j = self.goal_state
        grid[goal_i, goal_j] = 1

        if values is not None:
            for i in range(self.size):
                for j in range(self.size):
                    state = (i, j)
                    if state not in self.obstacles and state != self.goal_state:
                        grid[i, j] = values.get(state, 0)

        im = ax.imshow(grid, cmap="RdYlGn", aspect="equal")

        for i in range(self.size):
            for j in range(self.size):
                state = (i, j)
                if state == self.goal_state:
                    ax.text(
                        j,
                        i,
                        "G",
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                    )
                elif state in self.obstacles:
                    ax.text(
                        j,
                        i,
                        "X",
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                    )
                elif state == self.start_state:
                    ax.text(
                        j,
                        i,
                        "S",
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                    )
                elif values is not None:
                    ax.text(
                        j,
                        i,
                        f"{values.get(state, 0):.2f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                    )

        if policy is not None:
            arrow_map = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
            for state, action in policy.items():
                if state not in self.obstacles and state != self.goal_state:
                    i, j = state
                    if action in arrow_map:
                        ax.text(
                            j,
                            i - 0.3,
                            arrow_map[action],
                            ha="center",
                            va="center",
                            fontsize=12,
                            fontweight="bold",
                            color="blue",
                        )

        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.colorbar(im)
        plt.tight_layout()
        plt.show()


def create_custom_environment(
    size=4, goal_reward=10, step_reward=-0.1, obstacle_reward=-5, obstacles=None
):
    """Create a custom GridWorld environment"""
    env = GridWorld(size, goal_reward, step_reward, obstacle_reward)
    if obstacles:
        env.obstacles = obstacles
        env._build_transition_model()
    return env
