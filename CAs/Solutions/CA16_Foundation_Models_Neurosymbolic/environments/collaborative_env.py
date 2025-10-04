"""
Collaborative Environment for Human-AI Interaction

This module implements environments that support human-AI collaboration.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from gymnasium import Env
from gymnasium.spaces import Discrete, Box


class CollaborativeGridWorld(Env):
    """Grid world environment that supports human-AI collaboration."""

    def __init__(
        self,
        size: int = 8,
        num_goals: int = 3,
        num_obstacles: int = 5,
        collaboration_mode: bool = True,
        human_assistance_prob: float = 0.3,
    ):
        super().__init__()
        self.size = size
        self.num_goals = num_goals
        self.num_obstacles = num_obstacles
        self.collaboration_mode = collaboration_mode
        self.human_assistance_prob = human_assistance_prob

        # Action space: 0=up, 1=down, 2=left, 3=right, 4=human_help
        self.action_space = Discrete(5)

        # Observation space: flattened grid + agent position + goal positions + collaboration info
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(size * size + 2 + num_goals * 2 + 2,),
            dtype=np.float32,
        )

        # Environment state
        self.agent_pos = None
        self.goals = None
        self.obstacles = None
        self.grid = None

        # Collaboration state
        self.human_available = True
        self.human_assistance_count = 0
        self.ai_confidence = 0.5
        self.collaboration_history = []

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

        # Reset collaboration state
        self.human_assistance_count = 0
        self.ai_confidence = 0.5
        self.collaboration_history.clear()

        # Reset episode tracking
        self.episode_length = 0

        return self.get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        info = {"human_assistance": False, "ai_confidence": self.ai_confidence}

        # Check if human assistance is requested
        if action == 4:  # Human help action
            if self.human_available and np.random.random() < self.human_assistance_prob:
                # Human provides assistance
                self.human_assistance_count += 1
                info["human_assistance"] = True
                info["human_guidance"] = self._get_human_guidance()

                # Update AI confidence based on human assistance
                self.ai_confidence = min(1.0, self.ai_confidence + 0.1)

                # Human suggests a better action
                suggested_action = self._get_human_suggested_action()
                action = suggested_action
            else:
                # Human not available, use AI action
                action = self._get_ai_action()
                self.ai_confidence = max(0.0, self.ai_confidence - 0.05)

        # Execute the action
        if action < 4:  # Valid movement action
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
                info["collision"] = True
            else:
                self.agent_pos = new_pos

                # Check for goals
                if new_pos in self.goals:
                    self.goals.remove(new_pos)
                    reward = 10
                    info["goal_reached"] = True
                else:
                    reward = -0.1

                done = len(self.goals) == 0

        else:
            # Invalid action
            reward = -1
            done = False
            info["invalid_action"] = True

        # Update episode length
        self.episode_length += 1

        # Check for episode termination
        if self.episode_length >= self.max_episode_length:
            done = True

        # Record collaboration history
        self.collaboration_history.append(
            {
                "action": action,
                "human_assistance": info["human_assistance"],
                "ai_confidence": self.ai_confidence,
                "reward": reward,
            }
        )

        return self.get_observation(), reward, done, False, info

    def get_observation(self) -> np.ndarray:
        """Get current observation including collaboration info."""
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

        # Collaboration info
        collaboration_info = np.array(
            [
                self.ai_confidence,
                1.0 if self.human_available else 0.0,
            ],
            dtype=np.float32,
        )

        # Combine all observations
        observation = np.concatenate(
            [grid_flat, agent_pos, goal_positions, collaboration_info]
        )

        return observation.astype(np.float32)

    def _get_human_guidance(self) -> str:
        """Get human guidance for current situation."""
        # Simple heuristic for human guidance
        if self.agent_pos in self.goals:
            return "You're at a goal! Good job!"
        elif any(
            abs(self.agent_pos[0] - goal[0]) + abs(self.agent_pos[1] - goal[1]) <= 1
            for goal in self.goals
        ):
            return "You're close to a goal, keep going!"
        elif any(
            abs(self.agent_pos[0] - obs[0]) + abs(self.agent_pos[1] - obs[1]) <= 1
            for obs in self.obstacles
        ):
            return "Watch out for obstacles nearby!"
        else:
            return "Explore the environment to find goals!"

    def _get_human_suggested_action(self) -> int:
        """Get human-suggested action based on current state."""
        # Simple heuristic for human suggestions
        if not self.goals:
            return np.random.randint(0, 4)  # Random if no goals

        # Find closest goal
        closest_goal = min(
            self.goals,
            key=lambda g: abs(self.agent_pos[0] - g[0]) + abs(self.agent_pos[1] - g[1]),
        )

        # Suggest action towards closest goal
        if closest_goal[0] > self.agent_pos[0]:  # Goal is below
            return 1  # down
        elif closest_goal[0] < self.agent_pos[0]:  # Goal is above
            return 0  # up
        elif closest_goal[1] > self.agent_pos[1]:  # Goal is to the right
            return 3  # right
        else:  # Goal is to the left
            return 2  # left

    def _get_ai_action(self) -> int:
        """Get AI action when human is not available."""
        # Simple AI behavior
        return np.random.randint(0, 4)

    def set_human_availability(self, available: bool):
        """Set human availability."""
        self.human_available = available

    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """Get collaboration statistics."""
        if not self.collaboration_history:
            return {"total_interactions": 0}

        total_interactions = len(self.collaboration_history)
        human_assistances = sum(
            1 for h in self.collaboration_history if h["human_assistance"]
        )

        return {
            "total_interactions": total_interactions,
            "human_assistances": human_assistances,
            "human_assistance_rate": (
                human_assistances / total_interactions if total_interactions > 0 else 0
            ),
            "ai_confidence": self.ai_confidence,
            "human_assistance_count": self.human_assistance_count,
        }

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
            print(f"Agent: {self.agent_pos}, Goals: {self.goals}")
            print(
                f"AI Confidence: {self.ai_confidence:.2f}, Human Available: {self.human_available}"
            )
            print(
                f"Human Assists: {self.human_assistance_count}, Episode: {self.episode_length}"
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
