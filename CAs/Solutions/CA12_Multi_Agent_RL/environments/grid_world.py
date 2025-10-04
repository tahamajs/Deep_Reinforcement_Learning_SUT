"""
Grid world environment implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional


class GridWorldEnvironment:
    """Grid world environment for multi-agent coordination."""

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        n_agents: int = 2,
        n_targets: int = 3,
        max_steps: int = 200,
    ):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.max_steps = max_steps

        # Grid state: 0=empty, 1=agent, 2=target, 3=obstacle
        self.grid = np.zeros((height, width), dtype=int)
        self.agent_positions = []
        self.target_positions = []
        self.current_step = 0

        # Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.action_spaces = [spaces.Discrete(5) for _ in range(n_agents)]

        # Observation space: flattened grid + agent positions
        obs_dim = width * height + n_agents * 2
        self.observation_spaces = [
            spaces.Box(0, 1, (obs_dim,), dtype=np.float32) for _ in range(n_agents)
        ]

    def reset(self) -> List[np.ndarray]:
        """Reset environment."""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.current_step = 0

        # Place agents randomly
        self.agent_positions = []
        for i in range(self.n_agents):
            while True:
                pos = (np.random.randint(0, self.height), np.random.randint(0, self.width))
                if pos not in self.agent_positions:
                    self.agent_positions.append(pos)
                    break

        # Place targets randomly
        self.target_positions = []
        for i in range(self.n_targets):
            while True:
                pos = (np.random.randint(0, self.height), np.random.randint(0, self.width))
                if pos not in self.agent_positions and pos not in self.target_positions:
                    self.target_positions.append(pos)
                    break

        # Update grid
        self._update_grid()

        return self._get_observations()

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict[str, Any]]:
        """Execute actions."""
        rewards = [0.0] * self.n_agents

        # Move agents
        for i, action in enumerate(actions):
            new_pos = self._get_new_position(self.agent_positions[i], action)

            # Check if new position is valid
            if (
                0 <= new_pos[0] < self.height
                and 0 <= new_pos[1] < self.width
                and self.grid[new_pos] == 0
            ):
                self.agent_positions[i] = new_pos

        # Update grid
        self._update_grid()

        # Check for target collection
        for i, agent_pos in enumerate(self.agent_positions):
            if agent_pos in self.target_positions:
                rewards[i] += 10.0  # Reward for collecting target
                self.target_positions.remove(agent_pos)

        # Cooperation bonus
        if len(self.target_positions) == 0:
            rewards = [r + 50.0 for r in rewards]  # Team bonus

        self.current_step += 1
        done = len(self.target_positions) == 0 or self.current_step >= self.max_steps

        observations = self._get_observations()
        info = {
            "step": self.current_step,
            "targets_remaining": len(self.target_positions),
        }

        return observations, rewards, done, info

    def _get_new_position(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get new position after action."""
        y, x = pos
        if action == 0:  # up
            return (max(0, y - 1), x)
        elif action == 1:  # down
            return (min(self.height - 1, y + 1), x)
        elif action == 2:  # left
            return (y, max(0, x - 1))
        elif action == 3:  # right
            return (y, min(self.width - 1, x + 1))
        else:  # stay
            return (y, x)

    def _update_grid(self):
        """Update grid representation."""
        self.grid = np.zeros((self.height, self.width), dtype=int)

        # Place targets
        for pos in self.target_positions:
            self.grid[pos] = 2

        # Place agents
        for i, pos in enumerate(self.agent_positions):
            self.grid[pos] = 1

    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all agents."""
        observations = []
        flattened_grid = self.grid.flatten() / 2.0  # Normalize to [0, 1]

        for i in range(self.n_agents):
            # Include agent position in observation
            agent_pos = np.array(self.agent_positions[i]) / np.array(
                [self.height, self.width]
            )
            obs = np.concatenate([flattened_grid, agent_pos])

            # Pad with other agent positions
            other_agents = []
            for j in range(self.n_agents):
                if j != i:
                    other_pos = np.array(self.agent_positions[j]) / np.array(
                        [self.height, self.width]
                    )
                    other_agents.extend(other_pos)

            # Pad with zeros if needed
            while len(other_agents) < (self.n_agents - 1) * 2:
                other_agents.extend([0.0, 0.0])

            obs = np.concatenate([obs, other_agents[: 2 * (self.n_agents - 1)]])
            observations.append(obs.astype(np.float32))

        return observations

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the grid world."""
        if mode == "human":
            fig, ax = plt.subplots(figsize=(8, 8))

            # Create grid visualization
            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y, x] == 1:  # Agent
                        circle = patches.Circle(
                            (x + 0.5, y + 0.5), 0.3, facecolor="blue", edgecolor="black"
                        )
                        ax.add_patch(circle)
                    elif self.grid[y, x] == 2:  # Target
                        circle = patches.Circle(
                            (x + 0.5, y + 0.5),
                            0.3,
                            facecolor="green",
                            edgecolor="black",
                        )
                        ax.add_patch(circle)

            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect("equal")
            ax.grid(True)
            ax.set_title(f"Grid World (Step {self.current_step})")
            ax.invert_yaxis()  # Invert y-axis to match matrix indexing

            plt.show()

        return None
