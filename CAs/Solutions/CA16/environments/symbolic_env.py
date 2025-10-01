"""
Symbolic Environment for Neurosymbolic RL

This module implements a symbolic grid world environment for testing neurosymbolic RL.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from gymnasium import Env
from gymnasium.spaces import Discrete, Box


class SymbolicGridWorld(Env):
    """Symbolic grid world environment for neurosymbolic RL."""

    def __init__(self, size: int = 8, num_goals: int = 3, num_obstacles: int = 5):
        super().__init__()
        self.size = size
        self.num_goals = num_goals
        self.num_obstacles = num_obstacles

        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = Discrete(4)

        # Observation space: flattened grid + agent position + goal positions
        self.observation_space = Box(
            low=0, high=1, shape=(size * size + 2 + num_goals * 2,), dtype=np.float32
        )

        # Environment state
        self.agent_pos = None
        self.goals = None
        self.obstacles = None
        self.grid = None

        # Symbolic knowledge
        self.symbolic_rules = {
            "near_goal": lambda pos, goals: any(
                abs(pos[0] - g[0]) + abs(pos[1] - g[1]) <= 1 for g in goals
            ),
            "near_obstacle": lambda pos, obstacles: any(
                abs(pos[0] - o[0]) + abs(pos[1] - o[1]) <= 1 for o in obstacles
            ),
            "at_goal": lambda pos, goals: pos in goals,
            "at_obstacle": lambda pos, obstacles: pos in obstacles,
            "can_move_up": lambda pos, obstacles: pos[0] > 0
            and (pos[0] - 1, pos[1]) not in obstacles,
            "can_move_down": lambda pos, obstacles: pos[0] < self.size - 1
            and (pos[0] + 1, pos[1]) not in obstacles,
            "can_move_left": lambda pos, obstacles: pos[1] > 0
            and (pos[0], pos[1] - 1) not in obstacles,
            "can_move_right": lambda pos, obstacles: pos[1] < self.size - 1
            and (pos[0], pos[1] + 1) not in obstacles,
        }

        # Episode tracking
        self.episode_length = 0
        self.max_episode_length = 200

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Initialize agent position
        self.agent_pos = [0, 0]

        # Initialize goals
        self.goals = []
        while len(self.goals) < self.num_goals:
            goal = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            if goal != self.agent_pos and goal not in self.goals:
                self.goals.append(goal)

        # Initialize obstacles
        self.obstacles = []
        while len(self.obstacles) < self.num_obstacles:
            obstacle = [
                np.random.randint(0, self.size),
                np.random.randint(0, self.size),
            ]
            if (
                obstacle != self.agent_pos
                and obstacle not in self.goals
                and obstacle not in self.obstacles
            ):
                self.obstacles.append(obstacle)

        # Initialize grid
        self.grid = np.zeros((self.size, self.size))
        for goal in self.goals:
            self.grid[goal[0], goal[1]] = 2  # Goal
        for obstacle in self.obstacles:
            self.grid[obstacle[0], obstacle[1]] = 1  # Obstacle

        # Reset episode tracking
        self.episode_length = 0

        return self.get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Actions: 0=up, 1=down, 2=left, 3=right
        new_pos = self.agent_pos.copy()

        if action == 0:  # up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # down
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)
        elif action == 2:  # left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # right
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)

        # Check for obstacles
        if new_pos in self.obstacles:
            reward = -1
            done = False
        else:
            self.agent_pos = new_pos

            # Check for goals
            if new_pos in self.goals:
                self.goals.remove(new_pos)
                reward = 10
            else:
                reward = -0.1

            done = len(self.goals) == 0

        # Update episode length
        self.episode_length += 1

        # Check for episode termination
        if self.episode_length >= self.max_episode_length:
            done = True

        return self.get_observation(), reward, done, False, {}

    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Flatten grid
        grid_flat = self.grid.flatten()

        # Agent position
        agent_pos = np.array(self.agent_pos, dtype=np.float32) / self.size

        # Goal positions
        goal_positions = np.zeros(self.num_goals * 2, dtype=np.float32)
        for i, goal in enumerate(self.goals):
            if i < self.num_goals:
                goal_positions[i * 2] = goal[0] / self.size
                goal_positions[i * 2 + 1] = goal[1] / self.size

        # Combine all observations
        observation = np.concatenate([grid_flat, agent_pos, goal_positions])

        return observation.astype(np.float32)

    def get_symbolic_state(self) -> Dict[str, bool]:
        """Get symbolic representation of current state."""
        symbolic_state = {}

        for rule_name, rule_func in self.symbolic_rules.items():
            if "move" in rule_name:
                symbolic_state[rule_name] = rule_func(self.agent_pos, self.obstacles)
            else:
                symbolic_state[rule_name] = rule_func(self.agent_pos, self.goals)

        return symbolic_state

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            # Create display grid
            display_grid = np.zeros((self.size, self.size), dtype=str)
            display_grid.fill(".")

            # Place obstacles
            for obstacle in self.obstacles:
                display_grid[obstacle[0], obstacle[1]] = "X"

            # Place goals
            for goal in self.goals:
                display_grid[goal[0], goal[1]] = "G"

            # Place agent
            display_grid[self.agent_pos[0], self.agent_pos[1]] = "A"

            # Print grid
            print("\n" + "=" * (self.size * 2 + 1))
            for row in display_grid:
                print("|" + " ".join(row) + "|")
            print("=" * (self.size * 2 + 1))
            print(
                f"Agent: {self.agent_pos}, Goals: {self.goals}, Episode: {self.episode_length}"
            )

        elif mode == "rgb_array":
            # Create RGB array
            rgb_array = np.zeros((self.size, self.size, 3), dtype=np.uint8)

            # Set colors
            rgb_array[:, :] = [255, 255, 255]  # White background

            # Obstacles (black)
            for obstacle in self.obstacles:
                rgb_array[obstacle[0], obstacle[1]] = [0, 0, 0]

            # Goals (green)
            for goal in self.goals:
                rgb_array[goal[0], goal[1]] = [0, 255, 0]

            # Agent (red)
            rgb_array[self.agent_pos[0], self.agent_pos[1]] = [255, 0, 0]

            return rgb_array

        return None

    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            "size": self.size,
            "num_goals": self.num_goals,
            "num_obstacles": self.num_obstacles,
            "agent_pos": self.agent_pos,
            "goals": self.goals,
            "obstacles": self.obstacles,
            "episode_length": self.episode_length,
            "symbolic_rules": list(self.symbolic_rules.keys()),
        }

    def apply_symbolic_action(
        self, symbolic_action: str
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Apply a symbolic action."""
        # Map symbolic actions to numeric actions
        symbolic_to_numeric = {
            "move_up": 0,
            "move_down": 1,
            "move_left": 2,
            "move_right": 3,
        }

        if symbolic_action in symbolic_to_numeric:
            return self.step(symbolic_to_numeric[symbolic_action])
        else:
            # Invalid symbolic action
            return (
                self.get_observation(),
                -1,
                False,
                False,
                {"error": "Invalid symbolic action"},
            )

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions from current state."""
        valid_actions = []

        # Check each direction
        if self.symbolic_rules["can_move_up"](self.agent_pos, self.obstacles):
            valid_actions.append(0)
        if self.symbolic_rules["can_move_down"](self.agent_pos, self.obstacles):
            valid_actions.append(1)
        if self.symbolic_rules["can_move_left"](self.agent_pos, self.obstacles):
            valid_actions.append(2)
        if self.symbolic_rules["can_move_right"](self.agent_pos, self.obstacles):
            valid_actions.append(3)

        return valid_actions

    def get_symbolic_reward(self, action: int) -> float:
        """Get symbolic reward for an action."""
        # Get symbolic state
        symbolic_state = self.get_symbolic_state()

        # Base reward
        reward = -0.1

        # Reward for being near goals
        if symbolic_state["near_goal"]:
            reward += 0.5

        # Penalty for being near obstacles
        if symbolic_state["near_obstacle"]:
            reward -= 0.3

        # Reward for reaching goals
        if symbolic_state["at_goal"]:
            reward += 10

        # Penalty for hitting obstacles
        if symbolic_state["at_obstacle"]:
            reward -= 1

        return reward
