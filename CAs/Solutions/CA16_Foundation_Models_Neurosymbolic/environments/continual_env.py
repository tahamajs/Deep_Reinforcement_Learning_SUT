"""
Continual Learning Environments

This module contains environments designed for continual learning scenarios,
including task switching and domain adaptation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from typing import Dict, List, Any, Optional, Tuple
import random
import time
from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Configuration for a learning task."""

    task_id: int
    task_name: str
    difficulty: float
    reward_structure: Dict[str, float]
    state_distribution: str
    action_requirements: List[int]
    success_criteria: Dict[str, Any]


class ContinualLearningEnvironment(Env):
    """Environment for continual learning scenarios."""

    def __init__(self, num_tasks: int = 5, task_switch_frequency: int = 1000):
        super().__init__()

        self.num_tasks = num_tasks
        self.task_switch_frequency = task_switch_frequency

        # Action space: 0=up, 1=down, 2=left, 3=right, 4=special_action
        self.action_space = spaces.Discrete(5)

        # Observation space: position + task_id + task_features
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32
        )

        # Environment state
        self.current_task = 0
        self.task_step_count = 0
        self.agent_pos = np.array([0.0, 0.0])
        self.task_features = np.zeros(4)

        # Task configurations
        self.task_configs = self._create_task_configs()

        # Performance tracking
        self.task_performances = {}
        self.forgetting_measures = {}

        # Episode tracking
        self.episode_length = 0
        self.max_episode_length = 200

    def _create_task_configs(self) -> List[TaskConfig]:
        """Create task configurations."""
        configs = []

        for i in range(self.num_tasks):
            config = TaskConfig(
                task_id=i,
                task_name=f"task_{i}",
                difficulty=0.2 + i * 0.1,
                reward_structure={
                    "goal_reward": 10.0,
                    "step_penalty": -0.1,
                    "collision_penalty": -1.0,
                },
                state_distribution="uniform",
                action_requirements=[0, 1, 2, 3],  # Basic movement
                success_criteria={"max_steps": 50, "success_rate": 0.8},
            )
            configs.append(config)

        return configs

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset agent position
        self.agent_pos = np.array([0.0, 0.0])

        # Reset task features
        self.task_features = np.random.uniform(-1, 1, 4)

        # Reset episode tracking
        self.episode_length = 0

        return self.get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Check for task switch
        if (
            self.task_step_count > 0
            and self.task_step_count % self.task_switch_frequency == 0
        ):
            self._switch_task()

        # Execute action
        reward = self._execute_action(action)

        # Update state
        self.task_step_count += 1
        self.episode_length += 1

        # Check termination
        done = self.episode_length >= self.max_episode_length

        # Create info
        info = {
            "current_task": self.current_task,
            "task_step_count": self.task_step_count,
            "task_performance": self._get_task_performance(),
        }

        return self.get_observation(), reward, done, False, info

    def _execute_action(self, action: int) -> float:
        """Execute the given action."""
        config = self.task_configs[self.current_task]

        # Basic movement actions
        if action == 0:  # up
            self.agent_pos[1] += 0.1
        elif action == 1:  # down
            self.agent_pos[1] -= 0.1
        elif action == 2:  # left
            self.agent_pos[0] -= 0.1
        elif action == 3:  # right
            self.agent_pos[0] += 0.1
        elif action == 4:  # special action
            return self._execute_special_action()

        # Keep agent in bounds
        self.agent_pos = np.clip(self.agent_pos, -1, 1)

        # Calculate reward
        reward = config.reward_structure["step_penalty"]

        # Check for goal (simplified)
        if np.linalg.norm(self.agent_pos) < 0.1:
            reward += config.reward_structure["goal_reward"]

        return reward

    def _execute_special_action(self) -> float:
        """Execute special action for current task."""
        config = self.task_configs[self.current_task]

        # Task-specific special actions
        if self.current_task == 0:
            # Task 0: Move towards center
            direction = -self.agent_pos
            self.agent_pos += 0.2 * direction / (np.linalg.norm(direction) + 1e-8)
            return 1.0
        elif self.current_task == 1:
            # Task 1: Move in circle
            angle = np.arctan2(self.agent_pos[1], self.agent_pos[0])
            new_angle = angle + 0.1
            radius = 0.5
            self.agent_pos = radius * np.array([np.cos(new_angle), np.sin(new_angle)])
            return 0.5
        else:
            # Default: random movement
            self.agent_pos += 0.1 * np.random.uniform(-1, 1, 2)
            return 0.0

    def _switch_task(self):
        """Switch to the next task."""
        old_task = self.current_task
        self.current_task = (self.current_task + 1) % self.num_tasks

        # Record task switch
        if old_task not in self.task_performances:
            self.task_performances[old_task] = []

        # Update task features
        self.task_features = np.random.uniform(-1, 1, 4)

    def _get_task_performance(self) -> float:
        """Get performance for current task."""
        if self.current_task not in self.task_performances:
            return 0.0

        performances = self.task_performances[self.current_task]
        return np.mean(performances) if performances else 0.0

    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.concatenate(
            [
                self.agent_pos,
                [self.current_task / self.num_tasks],
                self.task_features,
            ]
        )
        return obs.astype(np.float32)

    def get_task_info(self) -> Dict[str, Any]:
        """Get information about current task."""
        config = self.task_configs[self.current_task]
        return {
            "task_id": config.task_id,
            "task_name": config.task_name,
            "difficulty": config.difficulty,
            "step_count": self.task_step_count,
        }


class TaskSwitchingEnvironment(Env):
    """Environment with explicit task switching."""

    def __init__(self, num_tasks: int = 3, switch_probability: float = 0.1):
        super().__init__()

        self.num_tasks = num_tasks
        self.switch_probability = switch_probability

        # Action space: 0=up, 1=down, 2=left, 3=right, 4=switch_task
        self.action_space = spaces.Discrete(5)

        # Observation space: position + task_id + task_specific_features
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(10,), dtype=np.float32
        )

        # Environment state
        self.current_task = 0
        self.agent_pos = np.array([0.0, 0.0])
        self.task_goals = {}
        self.task_obstacles = {}

        # Initialize tasks
        self._initialize_tasks()

        # Performance tracking
        self.task_switches = 0
        self.task_successes = {}
        self.task_attempts = {}

        # Episode tracking
        self.episode_length = 0
        self.max_episode_length = 300

    def _initialize_tasks(self):
        """Initialize task-specific elements."""
        for i in range(self.num_tasks):
            # Random goal for each task
            self.task_goals[i] = np.random.uniform(-0.8, 0.8, 2)

            # Random obstacles for each task
            num_obstacles = random.randint(2, 5)
            self.task_obstacles[i] = []
            for _ in range(num_obstacles):
                obstacle = np.random.uniform(-0.9, 0.9, 2)
                self.task_obstacles[i].append(obstacle)

            # Initialize performance tracking
            self.task_successes[i] = 0
            self.task_attempts[i] = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset agent position
        self.agent_pos = np.array([0.0, 0.0])

        # Reset current task
        self.current_task = 0

        # Reset episode tracking
        self.episode_length = 0

        return self.get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        reward = 0.0

        # Handle task switching
        if action == 4:  # switch task
            self._switch_task()
            reward += 0.1  # Small reward for task switching
        else:
            # Execute movement action
            reward += self._execute_movement(action)

        # Random task switching
        if random.random() < self.switch_probability:
            self._switch_task()

        # Update state
        self.episode_length += 1

        # Check termination
        done = self.episode_length >= self.max_episode_length

        # Create info
        info = {
            "current_task": self.current_task,
            "task_switches": self.task_switches,
            "task_success_rate": self._get_task_success_rate(),
        }

        return self.get_observation(), reward, done, False, info

    def _execute_movement(self, action: int) -> float:
        """Execute movement action."""
        old_pos = self.agent_pos.copy()

        # Movement
        if action == 0:  # up
            self.agent_pos[1] += 0.1
        elif action == 1:  # down
            self.agent_pos[1] -= 0.1
        elif action == 2:  # left
            self.agent_pos[0] -= 0.1
        elif action == 3:  # right
            self.agent_pos[0] += 0.1

        # Keep agent in bounds
        self.agent_pos = np.clip(self.agent_pos, -1, 1)

        # Check for collisions with obstacles
        for obstacle in self.task_obstacles[self.current_task]:
            if np.linalg.norm(self.agent_pos - obstacle) < 0.1:
                self.agent_pos = old_pos  # Revert movement
                return -1.0  # Collision penalty

        # Check for goal
        goal = self.task_goals[self.current_task]
        if np.linalg.norm(self.agent_pos - goal) < 0.1:
            self.task_successes[self.current_task] += 1
            return 10.0  # Goal reward

        # Step penalty
        return -0.01

    def _switch_task(self):
        """Switch to a random task."""
        old_task = self.current_task
        self.current_task = random.randint(0, self.num_tasks - 1)

        if self.current_task != old_task:
            self.task_switches += 1
            self.task_attempts[self.current_task] += 1

    def _get_task_success_rate(self) -> float:
        """Get overall task success rate."""
        total_successes = sum(self.task_successes.values())
        total_attempts = sum(self.task_attempts.values())

        return total_successes / max(1, total_attempts)

    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Agent position
        agent_obs = self.agent_pos

        # Task ID (one-hot encoded)
        task_obs = np.zeros(self.num_tasks)
        task_obs[self.current_task] = 1.0

        # Goal position
        goal_obs = self.task_goals[self.current_task]

        # Obstacle positions (flattened)
        obstacle_obs = np.zeros(6)  # Max 3 obstacles * 2 coordinates
        obstacles = self.task_obstacles[self.current_task]
        for i, obstacle in enumerate(obstacles[:3]):  # Limit to 3 obstacles
            obstacle_obs[i * 2 : (i + 1) * 2] = obstacle

        obs = np.concatenate([agent_obs, task_obs, goal_obs, obstacle_obs])
        return obs.astype(np.float32)

    def get_task_info(self) -> Dict[str, Any]:
        """Get information about current task."""
        return {
            "task_id": self.current_task,
            "goal_position": self.task_goals[self.current_task],
            "num_obstacles": len(self.task_obstacles[self.current_task]),
            "success_rate": self.task_successes[self.current_task]
            / max(1, self.task_attempts[self.current_task]),
        }
