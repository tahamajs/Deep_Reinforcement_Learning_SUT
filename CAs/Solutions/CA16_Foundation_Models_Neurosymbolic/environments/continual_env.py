"""
Continual Learning Environment

This module implements environments that support continual learning across multiple tasks.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from gymnasium import Env
from gymnasium.spaces import Discrete, Box


class ContinualLearningEnvironment(Env):
    """Environment that supports continual learning across multiple tasks."""

    def __init__(
        self,
        num_tasks: int = 5,
        state_dim: int = 4,
        action_dim: int = 2,
        task_switch_prob: float = 0.1,
        task_complexity_range: Tuple[float, float] = (0.1, 1.0),
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.task_switch_prob = task_switch_prob
        self.task_complexity_range = task_complexity_range

        # Action space
        self.action_space = Discrete(action_dim)

        # Observation space
        self.observation_space = Box(
            low=-1, high=1, shape=(state_dim + 1,), dtype=np.float32
        )  # +1 for task indicator

        # Task definitions
        self.tasks = self._generate_tasks()
        self.current_task = 0
        self.task_history = []

        # Environment state
        self.state = None
        self.episode_length = 0
        self.max_episode_length = 100

        # Continual learning metrics
        self.task_performances = {}
        self.forgetting_metrics = {}
        self.transfer_metrics = {}

    def _generate_tasks(self) -> List[Dict[str, Any]]:
        """Generate different tasks for continual learning."""
        tasks = []
        
        for task_id in range(self.num_tasks):
            # Generate task-specific parameters
            complexity = np.random.uniform(*self.task_complexity_range)
            
            # Task-specific reward function parameters
            reward_params = {
                "center": np.random.uniform(-1, 1, self.state_dim),
                "scale": np.random.uniform(0.1, 1.0),
                "complexity": complexity,
            }
            
            # Task-specific dynamics
            dynamics_params = {
                "drift": np.random.uniform(-0.1, 0.1, self.state_dim),
                "noise_scale": np.random.uniform(0.01, 0.1),
            }
            
            task = {
                "task_id": task_id,
                "reward_params": reward_params,
                "dynamics_params": dynamics_params,
                "complexity": complexity,
                "description": f"Task {task_id} (complexity: {complexity:.2f})",
            }
            
            tasks.append(task)
        
        return tasks

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Initialize state
        self.state = np.random.uniform(-1, 1, self.state_dim)

        # Reset episode tracking
        self.episode_length = 0

        # Record task start
        self.task_history.append({
            "task_id": self.current_task,
            "episode_start": True,
            "timestamp": len(self.task_history),
        })

        return self.get_observation(), {"task_id": self.current_task}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.state is None:
            raise ValueError("Environment not reset")

        # Get current task
        task = self.tasks[self.current_task]
        
        # Execute action
        next_state = self._apply_action(action, task)
        
        # Compute reward
        reward = self._compute_reward(next_state, task)
        
        # Check for task switch
        task_switched = False
        if np.random.random() < self.task_switch_prob:
            self._switch_task()
            task_switched = True

        # Update state
        self.state = next_state
        self.episode_length += 1

        # Check for episode termination
        done = self.episode_length >= self.max_episode_length

        # Create info dictionary
        info = {
            "task_id": self.current_task,
            "task_switched": task_switched,
            "episode_length": self.episode_length,
            "task_complexity": task["complexity"],
        }
        
        # Record step
        self.task_history.append({
            "task_id": self.current_task,
            "action": action,
            "reward": reward,
            "episode_length": self.episode_length,
            "timestamp": len(self.task_history),
        })

        return self.get_observation(), reward, done, False, info

    def _apply_action(self, action: int, task: Dict[str, Any]) -> np.ndarray:
        """Apply action to current state."""
        # Convert action to continuous action
        action_vector = np.zeros(self.state_dim)
        action_vector[action] = 1.0
        
        # Apply task-specific dynamics
        dynamics = task["dynamics_params"]
        next_state = self.state + action_vector * 0.1 + dynamics["drift"]
        
        # Add noise
        noise = np.random.normal(0, dynamics["noise_scale"], self.state_dim)
        next_state += noise
        
        # Clip to valid range
        next_state = np.clip(next_state, -1, 1)
        
        return next_state

    def _compute_reward(self, state: np.ndarray, task: Dict[str, Any]) -> float:
        """Compute reward for given state and task."""
        reward_params = task["reward_params"]
        
        # Distance-based reward
        distance = np.linalg.norm(state - reward_params["center"])
        reward = np.exp(-distance / reward_params["scale"])
        
        # Scale by task complexity
        reward *= reward_params["complexity"]
        
        return float(reward)

    def _switch_task(self):
        """Switch to a different task."""
        old_task = self.current_task
        self.current_task = (self.current_task + 1) % self.num_tasks

        # Record task switch
        self.task_history.append({
            "task_id": self.current_task,
            "task_switch": True,
            "from_task": old_task,
            "timestamp": len(self.task_history),
        })

    def get_observation(self) -> np.ndarray:
        """Get current observation including task indicator."""
        if self.state is None:
            return np.zeros(self.state_dim + 1)
        
        # Combine state and task indicator
        task_indicator = np.array([self.current_task / self.num_tasks])
        observation = np.concatenate([self.state, task_indicator])
        
        return observation.astype(np.float32)

    def set_task(self, task_id: int):
        """Set the current task."""
        if 0 <= task_id < self.num_tasks:
            self.current_task = task_id

    def get_task_info(self, task_id: int) -> Dict[str, Any]:
        """Get information about a specific task."""
        if 0 <= task_id < self.num_tasks:
            return self.tasks[task_id]
        return {}

    def get_task_performance(self, task_id: int) -> Dict[str, Any]:
        """Get performance metrics for a specific task."""
        if task_id not in self.task_performances:
            return {"avg_reward": 0.0, "episodes": 0}
        
        return self.task_performances[task_id]

    def update_task_performance(self, task_id: int, reward: float):
        """Update performance metrics for a task."""
        if task_id not in self.task_performances:
            self.task_performances[task_id] = {
                "rewards": [],
                "avg_reward": 0.0,
                "episodes": 0,
            }
        
        self.task_performances[task_id]["rewards"].append(reward)
        self.task_performances[task_id]["episodes"] += 1
        self.task_performances[task_id]["avg_reward"] = np.mean(
            self.task_performances[task_id]["rewards"]
        )

    def compute_forgetting_metrics(self) -> Dict[str, Any]:
        """Compute forgetting metrics across tasks."""
        if len(self.task_performances) < 2:
            return {"forgetting": 0.0, "retention": 1.0}
        
        forgetting_scores = []
        retention_scores = []
        
        for task_id in range(self.num_tasks):
            if task_id in self.task_performances:
                task_perf = self.task_performances[task_id]
                if len(task_perf["rewards"]) > 10:
                    # Use first 10 episodes as baseline
                    baseline_perf = np.mean(task_perf["rewards"][:10])
                    # Use last 10 episodes as current performance
                    current_perf = np.mean(task_perf["rewards"][-10:])
                    
                    forgetting = max(0, baseline_perf - current_perf)
                    retention = current_perf / baseline_perf if baseline_perf > 0 else 0
                    
                    forgetting_scores.append(forgetting)
                    retention_scores.append(retention)
        
        return {
            "forgetting": np.mean(forgetting_scores) if forgetting_scores else 0.0,
            "retention": np.mean(retention_scores) if retention_scores else 1.0,
            "task_forgetting": {
                task_id: forgetting_scores[i] if i < len(forgetting_scores) else 0.0
                for i, task_id in enumerate(self.task_performances.keys())
            },
        }

    def get_continual_learning_statistics(self) -> Dict[str, Any]:
        """Get continual learning statistics."""
        stats = {
            "num_tasks": self.num_tasks,
            "current_task": self.current_task,
            "task_performances": self.task_performances,
            "total_episodes": sum(
                perf["episodes"] for perf in self.task_performances.values()
            ),
        }
        
        # Add forgetting metrics
        forgetting_metrics = self.compute_forgetting_metrics()
        stats.update(forgetting_metrics)
        
        # Add task complexity analysis
        task_complexities = [task["complexity"] for task in self.tasks]
        stats["task_complexities"] = task_complexities
        stats["avg_complexity"] = np.mean(task_complexities)
        
        return stats

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            print(f"\nContinual Learning Environment")
            print(f"Current Task: {self.current_task}")
            print(f"Task Description: {self.tasks[self.current_task]['description']}")
            print(f"State: {self.state}")
            print(f"Episode Length: {self.episode_length}")
            print(f"Task Performances: {self.task_performances}")
        
        return None


class TaskSwitchingEnvironment(Env):
    """Environment that switches between different tasks during episodes."""

    def __init__(
        self,
        num_tasks: int = 3,
        state_dim: int = 4,
        action_dim: int = 2,
        switch_frequency: int = 20,
        task_overlap: float = 0.5,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.switch_frequency = switch_frequency
        self.task_overlap = task_overlap

        # Action space
        self.action_space = Discrete(action_dim)

        # Observation space
        self.observation_space = Box(
            low=-1, high=1, shape=(state_dim + 1,), dtype=np.float32
        )

        # Task definitions
        self.tasks = self._generate_overlapping_tasks()
        self.current_task = 0
        self.task_switch_count = 0

        # Environment state
        self.state = None
        self.episode_length = 0
        self.max_episode_length = 100

    def _generate_overlapping_tasks(self) -> List[Dict[str, Any]]:
        """Generate tasks with overlapping features."""
        tasks = []
        
        for task_id in range(self.num_tasks):
            # Create overlapping reward centers
            center = np.random.uniform(-1, 1, self.state_dim)
            
            # Add overlap with previous tasks
            if task_id > 0:
                overlap = self.tasks[task_id - 1]["reward_params"]["center"]
                center = (1 - self.task_overlap) * center + self.task_overlap * overlap
            
            task = {
                "task_id": task_id,
                "reward_params": {
                    "center": center,
                    "scale": np.random.uniform(0.1, 1.0),
                },
                "dynamics_params": {
                    "drift": np.random.uniform(-0.1, 0.1, self.state_dim),
                    "noise_scale": np.random.uniform(0.01, 0.1),
                },
            }
            
            tasks.append(task)
        
        return tasks

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Initialize state
        self.state = np.random.uniform(-1, 1, self.state_dim)

        # Reset episode tracking
        self.episode_length = 0
        self.task_switch_count = 0
        self.current_task = 0

        return self.get_observation(), {"task_id": self.current_task}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.state is None:
            raise ValueError("Environment not reset")

        # Check for task switch
        task_switched = False
        if self.episode_length > 0 and self.episode_length % self.switch_frequency == 0:
            self._switch_task()
            task_switched = True
            self.task_switch_count += 1

        # Get current task
        task = self.tasks[self.current_task]
        
        # Execute action
        next_state = self._apply_action(action, task)
        
        # Compute reward
        reward = self._compute_reward(next_state, task)

        # Update state
        self.state = next_state
        self.episode_length += 1

        # Check for episode termination
        done = self.episode_length >= self.max_episode_length

        # Create info dictionary
        info = {
            "task_id": self.current_task,
            "task_switched": task_switched,
            "episode_length": self.episode_length,
            "task_switch_count": self.task_switch_count,
        }

        return self.get_observation(), reward, done, False, info

    def _apply_action(self, action: int, task: Dict[str, Any]) -> np.ndarray:
        """Apply action to current state."""
        # Convert action to continuous action
        action_vector = np.zeros(self.state_dim)
        action_vector[action] = 1.0
        
        # Apply task-specific dynamics
        dynamics = task["dynamics_params"]
        next_state = self.state + action_vector * 0.1 + dynamics["drift"]
        
        # Add noise
        noise = np.random.normal(0, dynamics["noise_scale"], self.state_dim)
        next_state += noise
        
        # Clip to valid range
        next_state = np.clip(next_state, -1, 1)
        
        return next_state

    def _compute_reward(self, state: np.ndarray, task: Dict[str, Any]) -> float:
        """Compute reward for given state and task."""
        reward_params = task["reward_params"]
        
        # Distance-based reward
        distance = np.linalg.norm(state - reward_params["center"])
        reward = np.exp(-distance / reward_params["scale"])
        
        return float(reward)

    def _switch_task(self):
        """Switch to the next task."""
        self.current_task = (self.current_task + 1) % self.num_tasks

    def get_observation(self) -> np.ndarray:
        """Get current observation including task indicator."""
        if self.state is None:
            return np.zeros(self.state_dim + 1)
        
        # Combine state and task indicator
        task_indicator = np.array([self.current_task / self.num_tasks])
        observation = np.concatenate([self.state, task_indicator])
        
        return observation.astype(np.float32)

    def get_task_switch_statistics(self) -> Dict[str, Any]:
        """Get task switching statistics."""
        return {
            "num_tasks": self.num_tasks,
            "current_task": self.current_task,
            "task_switch_count": self.task_switch_count,
            "switch_frequency": self.switch_frequency,
            "task_overlap": self.task_overlap,
        }