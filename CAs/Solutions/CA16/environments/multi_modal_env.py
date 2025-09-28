"""
Multi-Modal Grid World Environment

This module provides a multi-modal grid world environment for testing advanced RL algorithms.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MultiModalGridWorld(gym.Env):
    """Multi-modal grid world with visual, symbolic, and language modalities."""

    def __init__(self, size: int = 10, num_objects: int = 5, max_steps: int = 100):
        super().__init__()

        self.size = size
        self.num_objects = num_objects
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Dict(
            {
                "visual": spaces.Box(
                    low=0, high=255, shape=(size, size, 3), dtype=np.uint8
                ),
                "symbolic": spaces.MultiDiscrete(
                    [size, size, num_objects + 1]
                ),  # position + object types
                "language": spaces.Text(max_length=100),  # natural language description
            }
        )

        self.agent_pos = None
        self.objects = {}  # position -> object_type
        self.inventory = []  # held objects
        self.step_count = 0

        self.object_types = ["key", "door", "treasure", "obstacle", "powerup"]
        self.object_colors = {
            "key": [255, 215, 0],  # gold
            "door": [139, 69, 19],  # brown
            "treasure": [255, 0, 255],  # magenta
            "obstacle": [128, 128, 128],  # gray
            "powerup": [0, 255, 255],  # cyan
        }

        self.rewards = {
            "move": -0.01,
            "pickup_key": 1.0,
            "pickup_treasure": 5.0,
            "pickup_powerup": 2.0,
            "use_key": 10.0,
            "hit_obstacle": -1.0,
            "timeout": -2.0,
        }

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        self.agent_pos = (np.random.randint(self.size), np.random.randint(self.size))

        self.objects = {}
        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        np.random.shuffle(positions)

        for i, obj_type in enumerate(
            np.random.choice(self.object_types, self.num_objects)
        ):
            self.objects[positions[i]] = obj_type

        self.inventory = []
        self.step_count = 0

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action in the environment."""
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = self.step_count >= self.max_steps

        if action < 4:  # Movement actions
            reward += self._move_agent(action)
        elif action == 4:  # Pickup
            reward += self._pickup_object()
        elif action == 5:  # Drop
            reward += self._drop_object()
        elif action == 6:  # Use
            reward += self._use_object()

        if self._check_win_condition():
            reward += 20.0
            terminated = True
        elif self._check_lose_condition():
            reward += self.rewards["timeout"]
            terminated = True

        reward += self.rewards["move"]

        observation = self._get_observation()
        info = {
            "inventory": self.inventory.copy(),
            "objects_remaining": len(self.objects),
            "step_count": self.step_count,
        }

        return observation, reward, terminated, truncated, info

    def _move_agent(self, direction: int) -> float:
        """Move agent in specified direction."""
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]  # up, down, left, right
        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            return self.rewards["hit_obstacle"]  # Hit wall

        if new_pos in self.objects and self.objects[new_pos] == "obstacle":
            return self.rewards["hit_obstacle"]

        self.agent_pos = new_pos
        return 0.0

    def _pickup_object(self) -> float:
        """Pickup object at current position."""
        if self.agent_pos in self.objects:
            obj_type = self.objects[self.agent_pos]

            if obj_type != "door":  # Can't pickup doors
                self.inventory.append(obj_type)
                del self.objects[self.agent_pos]

                if obj_type == "key":
                    return self.rewards["pickup_key"]
                elif obj_type == "treasure":
                    return self.rewards["pickup_treasure"]
                elif obj_type == "powerup":
                    return self.rewards["pickup_powerup"]

        return 0.0

    def _drop_object(self) -> float:
        """Drop held object."""
        if self.inventory:
            obj_type = self.inventory.pop()
            self.objects[self.agent_pos] = obj_type
        return 0.0

    def _use_object(self) -> float:
        """Use held object."""
        if "key" in self.inventory and self.agent_pos in self.objects:
            if self.objects[self.agent_pos] == "door":
                self.inventory.remove("key")
                del self.objects[self.agent_pos]
                return self.rewards["use_key"]

        if "powerup" in self.inventory:
            self.inventory.remove("powerup")
            return 1.0

        return 0.0

    def _check_win_condition(self) -> bool:
        """Check if agent has won."""
        return len([obj for obj in self.objects.values() if obj == "treasure"]) == 0

    def _check_lose_condition(self) -> bool:
        """Check if agent has lost."""
        return self.step_count >= self.max_steps

    def _get_observation(self) -> Dict:
        """Get multi-modal observation."""
        return {
            "visual": self._get_visual_observation(),
            "symbolic": self._get_symbolic_observation(),
            "language": self._get_language_description(),
        }

    def _get_visual_observation(self) -> np.ndarray:
        """Get visual observation as RGB image."""
        image = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        for pos, obj_type in self.objects.items():
            color = self.object_colors[obj_type]
            image[pos[0], pos[1]] = color

        image[self.agent_pos[0], self.agent_pos[1]] = [255, 255, 255]

        return image

    def _get_symbolic_observation(self) -> np.ndarray:
        """Get symbolic observation as discrete values."""
        agent_x, agent_y = self.agent_pos

        current_obj = 0  # 0 = empty
        if self.agent_pos in self.objects:
            obj_type = self.objects[self.agent_pos]
            current_obj = self.object_types.index(obj_type) + 1

        return np.array([agent_x, agent_y, current_obj])

    def _get_language_description(self) -> str:
        """Get natural language description of the state."""
        descriptions = []

        descriptions.append(
            f"You are at position ({self.agent_pos[0]}, {self.agent_pos[1]})."
        )

        nearby_objects = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
                if (
                    0 <= pos[0] < self.size
                    and 0 <= pos[1] < self.size
                    and pos in self.objects
                ):
                    obj_type = self.objects[pos]
                    nearby_objects.append(
                        f"{obj_type} to the {self._get_direction_name(dx, dy)}"
                    )

        if nearby_objects:
            descriptions.append(f"You see: {', '.join(nearby_objects)}.")
        else:
            descriptions.append("You see no objects nearby.")

        if self.inventory:
            descriptions.append(f"You are carrying: {', '.join(self.inventory)}.")
        else:
            descriptions.append("You are not carrying anything.")

        return " ".join(descriptions)

    def _get_direction_name(self, dx: int, dy: int) -> str:
        """Get direction name from offset."""
        if dx == -1 and dy == 0:
            return "west"
        elif dx == 1 and dy == 0:
            return "east"
        elif dx == 0 and dy == -1:
            return "north"
        elif dx == 0 and dy == 1:
            return "south"
        elif dx == -1 and dy == -1:
            return "northwest"
        elif dx == -1 and dy == 1:
            return "southwest"
        elif dx == 1 and dy == -1:
            return "northeast"
        elif dx == 1 and dy == 1:
            return "southeast"
        else:
            return "unknown"

    def render(self, mode="human"):
        """Render the environment."""
        if mode == "rgb_array":
            return self._get_visual_observation()

        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        for pos, obj_type in self.objects.items():
            symbol = obj_type[0].upper()  # First letter
            grid[pos[0]][pos[1]] = symbol

        grid[self.agent_pos[0]][self.agent_pos[1]] = "A"

        print("\n".join(" ".join(row) for row in grid))
        print(f"Inventory: {self.inventory}")
        print(f"Step: {self.step_count}")

    def close(self):
        """Close the environment."""
        pass


class MultiModalGridWorldWithMemory(MultiModalGridWorld):
    """Multi-modal grid world with memory capabilities."""

    def __init__(self, *args, memory_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_size = memory_size
        self.state_memory = []  # List of (state, action, reward) tuples
        self.episode_memory = []  # Current episode memory

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with memory tracking."""
        current_state = self._get_observation()

        observation, reward, terminated, truncated, info = super().step(action)

        self.episode_memory.append(
            {
                "state": current_state,
                "action": action,
                "reward": reward,
                "next_state": observation,
            }
        )

        if len(self.episode_memory) > self.memory_size:
            self.episode_memory.pop(0)

        info["memory"] = self.episode_memory.copy()
        info["memory_size"] = len(self.episode_memory)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset with memory clearing."""
        observation, info = super().reset(seed, options)

        if self.episode_memory:
            self.state_memory.append(self.episode_memory.copy())

        self.episode_memory = []

        if len(self.state_memory) > 100:  # Keep last 100 episodes
            self.state_memory.pop(0)

        info["long_term_memory"] = len(self.state_memory)
        return observation, info
