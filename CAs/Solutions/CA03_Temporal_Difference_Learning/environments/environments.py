import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict, deque
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")

print("Libraries imported successfully!")
print("Environment configured for Temporal Difference Learning")
print("Session 3: Ready to explore model-free reinforcement learning!")


class GridWorld:
    """
    GridWorld environment for demonstrating TD learning algorithms
    Modified from Session 2 to support episodic interaction
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

        self.current_state = self.start_state

    def reset(self):
        """Reset environment to start state"""
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        """
        Take action and return (next_state, reward, done, info)
        Compatible with standard RL environment interface
        """
        if self.is_terminal(self.current_state):
            return self.current_state, 0, True, {}

        dx, dy = self.action_effects[action]
        next_x, next_y = self.current_state[0] + dx, self.current_state[1] + dy

        if not (0 <= next_x < self.size and 0 <= next_y < self.size):
            next_state = self.current_state  # Stay in place
        else:
            next_state = (next_x, next_y)

        if next_state == self.goal_state:
            reward = self.goal_reward
        elif next_state in self.obstacles:
            reward = self.obstacle_reward
            next_state = self.current_state  # Can't move into obstacle
        else:
            reward = self.step_reward

        done = next_state == self.goal_state

        self.current_state = next_state

        return next_state, reward, done, {}

    def get_valid_actions(self, state):
        """Get valid actions from a state"""
        if self.is_terminal(state):
            return []
        return self.actions

    def is_terminal(self, state):
        """Check if state is terminal"""
        return state == self.goal_state

    def visualize_values(self, values, title="State Values", policy=None):
        """Visualize state values and optional policy"""
        grid = np.zeros((self.size, self.size))
        for i, j in self.obstacles:
            grid[i, j] = min(values.values()) - 1  # Make obstacles darker

        for i in range(self.size):
            for j in range(self.size):
                state = (i, j)
                if state not in self.obstacles:
                    grid[i, j] = values.get(state, 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid, cmap="RdYlGn", aspect="equal")

        arrow_map = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
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
                        color="darkgreen",
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
                        color="darkred",
                    )
                elif state == self.start_state:
                    ax.text(
                        j,
                        i - 0.3,
                        "S",
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color="blue",
                    )
                    ax.text(
                        j,
                        i + 0.2,
                        f"{values.get(state, 0):.1f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
                else:
                    ax.text(
                        j,
                        i,
                        f"{values.get(state, 0):.1f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                    )

                if policy and state in policy and not self.is_terminal(state):
                    action = policy[state]
                    if action in arrow_map:
                        ax.text(
                            j + 0.3,
                            i - 0.3,
                            arrow_map[action],
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="blue",
                        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()
