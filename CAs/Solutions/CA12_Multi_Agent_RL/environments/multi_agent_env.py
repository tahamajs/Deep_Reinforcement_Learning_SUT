"""
Multi-agent environment implementation.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


class MultiAgentEnvironment:
    """Generic multi-agent environment for testing algorithms."""

    def __init__(
        self,
        n_agents: int = 2,
        state_dim: int = 10,
        action_dim: int = 4,
        max_steps: int = 100,
        reward_scale: float = 1.0,
    ):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.reward_scale = reward_scale

        # Initialize state
        self.current_step = 0
        self.state = None

        # Action spaces
        self.action_spaces = [spaces.Discrete(action_dim) for _ in range(n_agents)]

        # State bounds
        self.state_low = -np.ones(state_dim)
        self.state_high = np.ones(state_dim)

    def reset(self) -> List[np.ndarray]:
        """Reset environment and return initial observations."""
        self.current_step = 0
        self.state = np.random.uniform(self.state_low, self.state_high, self.state_dim)

        # Each agent observes the full state (centralized training)
        observations = [self.state.copy() for _ in range(self.n_agents)]
        return observations

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict[str, Any]]:
        """Execute actions and return next observations, rewards, done, info."""
        if self.current_step >= self.max_steps:
            return (
                [self.state.copy() for _ in range(self.n_agents)],
                [0.0] * self.n_agents,
                True,
                {},
            )

        # Simple dynamics: state evolves based on actions
        action_effects = np.zeros(self.state_dim)
        for i, action in enumerate(actions):
            # Each agent's action affects different dimensions
            start_idx = i * (self.state_dim // self.n_agents)
            end_idx = min(start_idx + (self.state_dim // self.n_agents), self.state_dim)

            if end_idx > start_idx:
                action_effects[start_idx:end_idx] = (
                    action / self.action_dim - 0.5
                ) * 0.1

        # Add some noise
        noise = np.random.normal(0, 0.01, self.state_dim)
        self.state += action_effects + noise

        # Clip state to bounds
        self.state = np.clip(self.state, self.state_low, self.state_high)

        # Compute rewards (cooperative task)
        team_reward = self._compute_team_reward()
        rewards = [team_reward] * self.n_agents

        self.current_step += 1
        done = self.current_step >= self.max_steps

        observations = [self.state.copy() for _ in range(self.n_agents)]
        info = {"step": self.current_step, "team_reward": team_reward}

        return observations, rewards, done, info

    def _compute_team_reward(self) -> float:
        """Compute team reward based on current state."""
        # Reward for being close to origin (cooperative goal)
        distance_from_origin = np.linalg.norm(self.state)
        reward = self.reward_scale * (
            1.0 - distance_from_origin / np.sqrt(self.state_dim)
        )
        return reward

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            plt.figure(figsize=(8, 6))
            plt.plot(self.state)
            plt.title(f"Multi-Agent Environment State (Step {self.current_step})")
            plt.xlabel("State Dimension")
            plt.ylabel("State Value")
            plt.grid(True)
            plt.show()

        return None

    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()


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
                pos = (np.random.randint(self.height), np.random.randint(self.width))
                if pos not in self.agent_positions:
                    self.agent_positions.append(pos)
                    break

        # Place targets randomly
        self.target_positions = []
        for i in range(self.n_targets):
            while True:
                pos = (np.random.randint(self.height), np.random.randint(self.width))
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
            return (y - 1, x)
        elif action == 1:  # down
            return (y + 1, x)
        elif action == 2:  # left
            return (y, x - 1)
        elif action == 3:  # right
            return (y, x + 1)
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


class CoordinationEnvironment:
    """Environment requiring coordination between agents."""

    def __init__(self, n_agents: int = 3, coordination_threshold: float = 0.7):
        self.n_agents = n_agents
        self.coordination_threshold = coordination_threshold
        self.state_dim = n_agents * 2
        self.action_dim = 2  # Binary actions

        self.state = None
        self.current_step = 0
        self.max_steps = 50

    def reset(self) -> List[np.ndarray]:
        """Reset environment."""
        self.state = np.random.uniform(-1, 1, self.state_dim)
        self.current_step = 0

        observations = [self.state.copy() for _ in range(self.n_agents)]
        return observations

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict[str, Any]]:
        """Execute actions."""
        # Compute coordination level
        action_consensus = np.mean(actions)
        coordination_level = 1.0 - abs(action_consensus - 0.5) * 2

        # State evolution based on coordination
        coordination_bonus = coordination_level * 0.1
        self.state += np.random.normal(0, 0.05, self.state_dim) + coordination_bonus

        # Rewards based on coordination
        if coordination_level > self.coordination_threshold:
            rewards = [1.0] * self.n_agents  # High reward for good coordination
        else:
            rewards = [-0.1] * self.n_agents  # Penalty for poor coordination

        self.current_step += 1
        done = self.current_step >= self.max_steps

        observations = [self.state.copy() for _ in range(self.n_agents)]
        info = {
            "coordination_level": coordination_level,
            "action_consensus": action_consensus,
        }

        return observations, rewards, done, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render coordination environment."""
        if mode == "human":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot state evolution
            ax1.plot(self.state)
            ax1.set_title("Agent States")
            ax1.set_xlabel("State Dimension")
            ax1.set_ylabel("State Value")
            ax1.grid(True)

            # Plot coordination metric
            coordination_history = getattr(self, "coordination_history", [])
            if coordination_history:
                ax2.plot(coordination_history)
                ax2.axhline(
                    y=self.coordination_threshold,
                    color="r",
                    linestyle="--",
                    label=f"Threshold ({self.coordination_threshold})",
                )
                ax2.set_title("Coordination Level")
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Coordination")
                ax2.legend()
                ax2.grid(True)

            plt.tight_layout()
            plt.show()

        return None
