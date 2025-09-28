"""
Collaborative Grid World Environment

This module provides a collaborative environment for human-AI interaction.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import threading
import time
import queue


class CollaborativeGridWorld(gym.Env):
    """Collaborative grid world where human and AI agents work together."""

    def __init__(self, size: int = 12, num_agents: int = 2, max_steps: int = 100):
        super().__init__()

        self.size = size
        self.num_agents = num_agents  # AI agents
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(
            6
        )  # up, down, left, right, interact, communicate

        self.observation_space = spaces.Dict(
            {
                "ai_state": spaces.Dict(
                    {
                        "position": spaces.MultiDiscrete([size, size]),
                        "inventory": spaces.MultiBinary(5),  # Can hold up to 5 items
                        "last_communication": spaces.Text(max_length=50),
                    }
                ),
                "human_state": spaces.Dict(
                    {
                        "available": spaces.Discrete(2),  # 0=unavailable, 1=available
                        "last_action": spaces.Discrete(6),
                        "communication_history": spaces.Sequence(
                            spaces.Text(max_length=50)
                        ),
                    }
                ),
                "shared_state": spaces.Dict(
                    {
                        "grid_objects": spaces.MultiDiscrete(
                            [size, size, 4]
                        ),  # object types
                        "task_progress": spaces.Box(
                            low=0, high=1, shape=(3,)
                        ),  # progress on subtasks
                        "communication_channel": spaces.Text(max_length=100),
                    }
                ),
            }
        )

        self.ai_positions = []
        self.ai_inventories = []
        self.human_available = True
        self.human_last_action = 0
        self.step_count = 0

        self.grid_objects = {}  # position -> object_type
        self.task_progress = np.zeros(3)  # Progress on 3 subtasks
        self.communication_history = []

        self.object_types = ["resource_a", "resource_b", "tool", "obstacle"]

        self.communication_queue = queue.Queue()
        self.human_response_queue = queue.Queue()
        self.communication_thread = None
        self.is_communicating = False

    def reset(self, seed=None, options=None):
        """Reset the collaborative environment."""
        super().reset(seed=seed)

        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        np.random.shuffle(positions)

        self.ai_positions = positions[: self.num_agents]
        self.ai_inventories = [[] for _ in range(self.num_agents)]

        self.grid_objects = {}
        for i in range(self.num_agents * 2):  # Some objects per agent
            obj_type = np.random.choice(self.object_types)
            self.grid_objects[positions[self.num_agents + i]] = obj_type

        self.human_available = True
        self.human_last_action = 0
        self.step_count = 0
        self.task_progress = np.zeros(3)
        self.communication_history = []

        self._start_communication()

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute collaborative step."""
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = self.step_count >= self.max_steps

        ai_reward = self._execute_ai_action(action)
        reward += ai_reward

        human_action = self._get_human_action()
        if human_action is not None:
            human_reward = self._execute_human_action(human_action)
            reward += human_reward

        self._update_task_progress()

        self._process_communication()

        if self._check_task_completion():
            reward += 20.0
            terminated = True

        observation = self._get_observation()
        info = {
            "ai_reward": ai_reward,
            "human_action": human_action,
            "task_progress": self.task_progress.copy(),
            "communication_count": len(self.communication_history),
            "collaboration_score": self._calculate_collaboration_score(),
        }

        return observation, reward, terminated, truncated, info

    def _execute_ai_action(self, action: int) -> float:
        """Execute AI agent action."""
        reward = 0

        if action < 4:  # Movement
            reward += self._move_ai_agent(0, action)  # Agent 0 for now
        elif action == 4:  # Interact
            reward += self._ai_interact(0)
        elif action == 5:  # Communicate
            reward += self._ai_communicate(0)

        return reward

    def _move_ai_agent(self, agent_id: int, direction: int) -> float:
        """Move AI agent."""
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]
        current_pos = self.ai_positions[agent_id]
        new_pos = (current_pos[0] + dx, current_pos[1] + dy)

        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            return -0.1

        if new_pos in self.grid_objects and self.grid_objects[new_pos] == "obstacle":
            return -0.1

        self.ai_positions[agent_id] = new_pos

        if new_pos in self.grid_objects:
            obj_type = self.grid_objects[new_pos]
            if obj_type in ["resource_a", "resource_b"]:
                self.ai_inventories[agent_id].append(obj_type)
                del self.grid_objects[new_pos]
                return 1.0

        return 0.0

    def _ai_interact(self, agent_id: int) -> float:
        """AI agent interaction."""
        pos = self.ai_positions[agent_id]

        if pos in self.grid_objects:
            obj_type = self.grid_objects[pos]
            if obj_type == "tool" and len(self.ai_inventories[agent_id]) < 5:
                self.ai_inventories[agent_id].append(obj_type)
                del self.grid_objects[pos]
                return 0.5

        return 0.0

    def _ai_communicate(self, agent_id: int) -> float:
        """AI agent communication."""
        message = self._generate_ai_message(agent_id)

        self.communication_queue.put(
            {
                "sender": "ai",
                "agent_id": agent_id,
                "message": message,
                "timestamp": time.time(),
            }
        )

        return 0.1  # Small reward for communication

    def _generate_ai_message(self, agent_id: int) -> str:
        """Generate AI communication message."""
        pos = self.ai_positions[agent_id]
        inventory = self.ai_inventories[agent_id]

        messages = [
            f"I'm at position {pos} with inventory {inventory}",
            f"I see objects nearby: {self._get_nearby_objects(pos)}",
            f"My progress: {self.task_progress}",
            "Need help with resource collection",
            "Task almost complete!",
        ]

        return np.random.choice(messages)

    def _get_nearby_objects(self, position: Tuple[int, int]) -> List[str]:
        """Get objects near a position."""
        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_pos = (position[0] + dx, position[1] + dy)
                if check_pos in self.grid_objects:
                    nearby.append(self.grid_objects[check_pos])
        return nearby

    def _get_human_action(self) -> Optional[int]:
        """Get human action if available."""
        if not self.human_available:
            return None

        try:
            human_input = self.human_response_queue.get_nowait()
            self.human_last_action = human_input.get("action", 0)
            return self.human_last_action
        except queue.Empty:
            return None

    def _execute_human_action(self, action: int) -> float:
        """Execute human action (simplified - humans can teleport and collect)."""
        if action < 4:
            new_pos = (np.random.randint(self.size), np.random.randint(self.size))
            nearest_resource = self._find_nearest_resource(new_pos)
            if nearest_resource:
                self.ai_positions[0] = nearest_resource  # Help AI agent
                return 2.0
        elif action == 4:
            collected = 0
            for pos, obj in list(self.grid_objects.items()):
                if obj in ["resource_a", "resource_b"]:
                    self.ai_inventories[0].append(obj)
                    del self.grid_objects[pos]
                    collected += 1
                    if collected >= 3:  # Collect up to 3 items
                        break
            return collected * 1.5
        elif action == 5:
            return 0.5

        return 0.0

    def _find_nearest_resource(
        self, start_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Find nearest resource position."""
        min_dist = float("inf")
        nearest_pos = None

        for pos, obj in self.grid_objects.items():
            if obj in ["resource_a", "resource_b"]:
                dist = abs(pos[0] - start_pos[0]) + abs(pos[1] - start_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_pos = pos

        return nearest_pos

    def _update_task_progress(self):
        """Update task progress based on current state."""
        resource_a_count = sum(
            1 for inv in self.ai_inventories for item in inv if item == "resource_a"
        )
        self.task_progress[0] = min(resource_a_count / 5.0, 1.0)  # Need 5 resource_a

        resource_b_count = sum(
            1 for inv in self.ai_inventories for item in inv if item == "resource_b"
        )
        self.task_progress[1] = min(resource_b_count / 3.0, 1.0)  # Need 3 resource_b

        tool_count = sum(
            1 for inv in self.ai_inventories for item in inv if item == "tool"
        )
        self.task_progress[2] = min(tool_count / 2.0, 1.0)  # Need 2 tools

    def _check_task_completion(self) -> bool:
        """Check if all tasks are complete."""
        return np.all(self.task_progress >= 1.0)

    def _process_communication(self):
        """Process communication messages."""
        try:
            while True:
                message = self.communication_queue.get_nowait()
                self.communication_history.append(message)

                if len(self.communication_history) > 10:
                    self.communication_history.pop(0)

        except queue.Empty:
            pass

    def _calculate_collaboration_score(self) -> float:
        """Calculate collaboration effectiveness score."""
        comm_score = min(len(self.communication_history) / 10.0, 1.0)
        progress_score = np.mean(self.task_progress)
        human_score = 1.0 if self.human_available else 0.0

        return (comm_score + progress_score + human_score) / 3.0

    def _get_observation(self) -> Dict:
        """Get collaborative observation."""
        return {
            "ai_state": {
                "position": np.array(self.ai_positions[0]),  # Primary AI agent
                "inventory": np.array(
                    [
                        1 if item in self.ai_inventories[0] else 0
                        for item in self.object_types[:5]
                    ]
                ),
                "last_communication": (
                    self.communication_history[-1]["message"]
                    if self.communication_history
                    else ""
                ),
            },
            "human_state": {
                "available": int(self.human_available),
                "last_action": self.human_last_action,
                "communication_history": [
                    msg["message"] for msg in self.communication_history[-3:]
                ],
            },
            "shared_state": {
                "grid_objects": self._get_grid_object_array(),
                "task_progress": self.task_progress,
                "communication_channel": self._get_communication_summary(),
            },
        }

    def _get_grid_object_array(self) -> np.ndarray:
        """Get grid objects as array."""
        obj_array = np.zeros((self.size, self.size), dtype=int)

        for pos, obj_type in self.grid_objects.items():
            obj_id = self.object_types.index(obj_type) + 1
            obj_array[pos[0], pos[1]] = obj_id

        for i, pos in enumerate(self.ai_positions):
            obj_array[pos[0], pos[1]] = -1 - i  # Negative values for agents

        return obj_array

    def _get_communication_summary(self) -> str:
        """Get summary of recent communications."""
        if not self.communication_history:
            return "No recent communications"

        recent = self.communication_history[-3:]
        summary = "; ".join([f"{msg['sender']}: {msg['message']}" for msg in recent])
        return summary

    def _start_communication(self):
        """Start communication handling thread."""
        if (
            self.communication_thread is None
            or not self.communication_thread.is_alive()
        ):
            self.is_communicating = True
            self.communication_thread = threading.Thread(
                target=self._communication_handler
            )
            self.communication_thread.daemon = True
            self.communication_thread.start()

    def _communication_handler(self):
        """Handle communication processing."""
        while self.is_communicating:
            self._process_communication()
            time.sleep(0.1)

    def render(self, mode="human"):
        """Render the collaborative environment."""
        if mode == "rgb_array":
            return self._render_rgb()

        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        for pos, obj_type in self.grid_objects.items():
            symbol = obj_type[0].upper()
            grid[pos[0]][pos[1]] = symbol

        for i, pos in enumerate(self.ai_positions):
            if grid[pos[0]][pos[1]] == ".":
                grid[pos[0]][pos[1]] = f"A{i}"

        print("\n".join(" ".join(row) for row in grid))
        print(f"AI Inventories: {self.ai_inventories}")
        print(f"Task Progress: {self.task_progress}")
        print(f"Human Available: {self.human_available}")
        print(f"Communications: {len(self.communication_history)}")

    def _render_rgb(self) -> np.ndarray:
        """Render as RGB array."""
        image = np.zeros((self.size * 10, self.size * 10, 3), dtype=np.uint8)

        colors = {
            "resource_a": [255, 0, 0],  # Red
            "resource_b": [0, 255, 0],  # Green
            "tool": [0, 0, 255],  # Blue
            "obstacle": [128, 128, 128],  # Gray
            "ai_agent": [255, 255, 0],  # Yellow
        }

        for i in range(self.size):
            for j in range(self.size):
                color = [255, 255, 255]  # White background

                pos = (i, j)
                if pos in self.grid_objects:
                    obj_type = self.grid_objects[pos]
                    color = colors.get(obj_type, [128, 128, 128])

                for agent_id, agent_pos in enumerate(self.ai_positions):
                    if agent_pos == pos:
                        color = colors["ai_agent"]

                image[i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = color

        return image

    def close(self):
        """Close the environment."""
        self.is_communicating = False
        if self.communication_thread:
            self.communication_thread.join(timeout=2)
