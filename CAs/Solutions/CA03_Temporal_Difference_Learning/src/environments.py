"""
environments.py - GridWorld Environment for Reinforcement Learning.

This module defines the GridWorld environment, a classic reinforcement learning
testbed used for temporal difference learning experiments. It includes state
space definition, action dynamics, reward structure, and visualization utilities.
"""

from typing import Dict, Tuple, List, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GridWorld:
    """
    A simple grid-world environment for reinforcement learning.

    The environment is a square grid where an agent can move Up, Down, Left, Right.
    It contains a start state, a goal state, and optional obstacles.
    """

    def __init__(
        self,
        size: int = 4,
        start_state: Tuple[int, int] = (0, 0),
        goal_state: Tuple[int, int] = (3, 3),
        obstacles: List[Tuple[int, int]] = None,
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        obstacle_reward: float = -5.0,
    ):
        """
        Initializes the GridWorld environment.

        Args:
            size (int): The size of the N x N grid.
            start_state (Tuple[int, int]): The starting position of the agent.
            goal_state (Tuple[int, int]): The goal position.
            obstacles (List[Tuple[int, int]]): A list of obstacle positions.
            step_reward (float): Reward for each step taken.
            goal_reward (float): Reward for reaching the goal state.
            obstacle_reward (float): Reward for hitting an obstacle.
        """
        if not (0 <= start_state[0] < size and 0 <= start_state[1] < size):
            raise ValueError(f"Start state {start_state} is outside the grid of size {size}.")
        if not (0 <= goal_state[0] < size and 0 <= goal_state[1] < size):
            raise ValueError(f"Goal state {goal_state} is outside the grid of size {size}.")
        if obstacles:
            for obs in obstacles:
                if not (0 <= obs[0] < size and 0 <= obs[1] < size):
                    raise ValueError(f"Obstacle state {obs} is outside the grid of size {size}.")

        self.size = size
        self.states = [(r, c) for r in range(size) for c in range(size)]
        self.actions = ["up", "down", "left", "right"]
        self.start_state = start_state
        self.goal_state = goal_state
        self.obstacles = set(obstacles) if obstacles else set()
        self.current_state = self.start_state

        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.obstacle_reward = obstacle_reward

        self._validate_config()

    def _validate_config(self) -> None:
        """Validates the environment configuration to prevent invalid setups."""
        if self.start_state == self.goal_state:
            print("Warning: Start state is also the goal state. Consider adjusting for meaningful episodes.")
        if self.start_state in self.obstacles:
            raise ValueError(f"Start state {self.start_state} cannot be an obstacle.")
        if self.goal_state in self.obstacles:
            raise ValueError(f"Goal state {self.goal_state} cannot be an obstacle.")

    def reset(self) -> Tuple[int, int]:
        """
        Resets the environment to the initial state.

        Returns:
            Tuple[int, int]: The initial state.
        """
        self.current_state = self.start_state
        return self.current_state

    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        Takes a step in the environment given an action.

        Args:
            action (str): The action to take ("up", "down", "left", "right").

        Returns:
            Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
                - next_state (Tuple[int, int]): The new state after taking the action.
                - reward (float): The reward received.
                - done (bool): True if the episode has ended, False otherwise.
                - info (Dict[str, Any]): Additional information (empty for this environment).
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}. Must be one of {self.actions}")

        r, c = self.current_state
        next_r, next_c = r, c

        if action == "up":
            next_r -= 1
        elif action == "down":
            next_r += 1
        elif action == "left":
            next_c -= 1
        elif action == "right":
            next_c += 1

        # Check boundary conditions
        if not (0 <= next_r < self.size and 0 <= next_c < self.size):
            next_state = self.current_state  # Stay in the current state if boundary hit
            reward = self.obstacle_reward  # Penalize for hitting boundary
        else:
            next_state = (next_r, next_c)
            reward = self.step_reward

            if next_state in self.obstacles:
                reward = self.obstacle_reward
                next_state = self.current_state # Agent doesn't move if it hits an obstacle
            elif next_state == self.goal_state:
                reward = self.goal_reward
        
        self.current_state = next_state
        done = self.is_terminal(self.current_state)

        return self.current_state, reward, done, {}

    def get_valid_actions(self, state: Tuple[int, int]) -> List[str]:
        """
        Returns a list of valid actions from a given state.
        In this gridworld, all actions are always valid, though some might lead to staying in place.

        Args:
            state (Tuple[int, int]): The current state.

        Returns:
            List[str]: A list of possible actions.
        """
        return self.actions

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """
        Checks if a state is a terminal state (goal or obstacle).

        Args:
            state (Tuple[int, int]): The state to check.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        return state == self.goal_state or state in self.obstacles

    def visualize_values(
        self,
        values: Dict[Tuple[int, int], float],
        title: str = "GridWorld Value Function",
        policy: Dict[Tuple[int, int], str] = None,
        filepath: str = None
    ) -> None:
        """
        Visualizes the value function and optionally the policy on the grid.

        Args:
            values (Dict[Tuple[int, int], float]): A dictionary mapping states to their values.
            title (str): The title of the plot.
            policy (Dict[Tuple[int, int], str], optional): A dictionary mapping states to optimal actions.
                                                            Defaults to None.
            filepath (str, optional): Path to save the figure. If None, displays the figure.
        """
        grid_values = np.zeros((self.size, self.size))
        for r, c in self.states:
            grid_values[r, c] = values.get((r, c), 0.0)

        plt.figure(figsize=(self.size, self.size))
        sns.heatmap(
            grid_values,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar=True,
            linewidths=0.5,
            linecolor="black",
            yticklabels=False,
            xticklabels=False,
        )

        # Mark start, goal, and obstacles
        for r, c in self.states:
            if (r, c) == self.start_state:
                plt.text(c + 0.5, r + 0.5, "S", color="red", ha="center", va="center", fontsize=16)
            elif (r, c) == self.goal_state:
                plt.text(c + 0.5, r + 0.5, "G", color="green", ha="center", va="center", fontsize=16)
            elif (r, c) in self.obstacles:
                plt.text(c + 0.5, r + 0.5, "X", color="black", ha="center", va="center", fontsize=16)
            
            # Draw policy arrows
            if policy and (r, c) in policy and (r,c) != self.goal_state and (r,c) not in self.obstacles:
                action = policy[(r,c)]
                dx, dy = 0, 0
                if action == "up": dy = -0.3
                elif action == "down": dy = 0.3
                elif action == "left": dx = -0.3
                elif action == "right": dx = 0.3
                plt.arrow(c + 0.5, r + 0.5, dx, dy, head_width=0.2, head_length=0.2, fc='white', ec='white')

        plt.title(title, fontsize=16)
        plt.tight_layout()
        if filepath:
            plt.savefig(filepath)
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    # Example Usage:
    env = GridWorld()
    print("GridWorld Environment Configuration:")
    print(f"  • State space: {len(env.states)} states")
    print(f"  • Action space: {len(env.actions)} actions")
    print(f"  • Start state: {env.start_state}")
    print(f"  • Goal state: {env.goal_state}")
    print(f"  • Obstacles: {env.obstacles}")
    
    state = env.reset()
    print(f"\nEnvironment reset. Current state: {state}")
    
    next_state, reward, done, info = env.step("right")
    print(f"Action 'right': next_state={next_state}, reward={reward}, done={done}")

    # Visualize an empty value function
    env.visualize_values({state: 0 for state in env.states}, title="GridWorld Environment Layout")

    # Create a dummy policy for visualization
    dummy_policy = {}
    for r,c in env.states:
        if (r+c) % 2 == 0:
            dummy_policy[(r,c)] = "right"
        else:
            dummy_policy[(r,c)] = "down"

    env.visualize_values(
        {(s): np.random.rand() for s in env.states},
        title="GridWorld with Dummy Policy",
        policy=dummy_policy,
        filepath="visualizations/dummy_gridworld.png"
    )
    print("Dummy GridWorld visualization saved to visualizations/dummy_gridworld.png")


