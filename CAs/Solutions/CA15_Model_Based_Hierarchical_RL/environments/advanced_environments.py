"""
Advanced Complex Environments

This module contains complex environments for testing advanced RL algorithms:
- Multi-agent environments with cooperation and competition
- Continuous control environments with complex dynamics
- Hierarchical task environments with multiple subgoals
- Dynamic environments with changing goals and obstacles
- Realistic robotics simulation environments
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
import math
from collections import deque
import gym
from gym import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiAgentCooperationEnv:
    """Multi-agent environment requiring cooperation to achieve goals."""

    def __init__(self, num_agents=3, grid_size=10, num_goals=2):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_goals = num_goals

        # Agent positions
        self.agent_positions = np.zeros((num_agents, 2))
        self.goal_positions = np.zeros((num_goals, 2))

        # Action space: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0,
            high=grid_size - 1,
            shape=(num_agents * 2 + num_goals * 2,),
            dtype=np.float32,
        )

        # Cooperation requirements
        self.cooperation_threshold = 2  # Minimum agents needed at goal
        self.goal_rewards = [10, 15]  # Different rewards for different goals

        self.reset()

    def reset(self):
        """Reset environment."""
        # Randomly place agents
        for i in range(self.num_agents):
            self.agent_positions[i] = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            ]

        # Randomly place goals
        for i in range(self.num_goals):
            self.goal_positions[i] = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            ]

        return self._get_observation()

    def step(self, actions):
        """Take step with all agents."""
        rewards = np.zeros(self.num_agents)
        dones = np.zeros(self.num_agents, dtype=bool)

        # Move agents
        for i, action in enumerate(actions):
            if action == 0:  # Up
                self.agent_positions[i][1] = min(
                    self.grid_size - 1, self.agent_positions[i][1] + 1
                )
            elif action == 1:  # Down
                self.agent_positions[i][1] = max(0, self.agent_positions[i][1] - 1)
            elif action == 2:  # Left
                self.agent_positions[i][0] = max(0, self.agent_positions[i][0] - 1)
            elif action == 3:  # Right
                self.agent_positions[i][0] = min(
                    self.grid_size - 1, self.agent_positions[i][0] + 1
                )
            # action == 4: Stay

        # Check goal achievements
        for goal_idx in range(self.num_goals):
            agents_at_goal = 0
            for agent_idx in range(self.num_agents):
                if np.allclose(
                    self.agent_positions[agent_idx], self.goal_positions[goal_idx]
                ):
                    agents_at_goal += 1

            if agents_at_goal >= self.cooperation_threshold:
                # Reward all agents at this goal
                for agent_idx in range(self.num_agents):
                    if np.allclose(
                        self.agent_positions[agent_idx], self.goal_positions[goal_idx]
                    ):
                        rewards[agent_idx] += self.goal_rewards[goal_idx]

        # Individual rewards for proximity to goals
        for agent_idx in range(self.num_agents):
            min_distance = float("inf")
            for goal_idx in range(self.num_goals):
                distance = np.linalg.norm(
                    self.agent_positions[agent_idx] - self.goal_positions[goal_idx]
                )
                min_distance = min(min_distance, distance)
            rewards[agent_idx] += 1.0 / (1.0 + min_distance)  # Proximity reward

        # Check if all goals achieved
        all_goals_achieved = True
        for goal_idx in range(self.num_goals):
            agents_at_goal = 0
            for agent_idx in range(self.num_agents):
                if np.allclose(
                    self.agent_positions[agent_idx], self.goal_positions[goal_idx]
                ):
                    agents_at_goal += 1
            if agents_at_goal < self.cooperation_threshold:
                all_goals_achieved = False
                break

        if all_goals_achieved:
            dones[:] = True

        return self._get_observation(), rewards, dones, {}

    def _get_observation(self):
        """Get observation for all agents."""
        obs = np.concatenate(
            [self.agent_positions.flatten(), self.goal_positions.flatten()]
        )
        return obs.astype(np.float32)

    def render(self):
        """Render environment."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color="black", linewidth=0.5)
            ax.axvline(i - 0.5, color="black", linewidth=0.5)

        # Draw agents
        colors = ["red", "blue", "green", "orange", "purple"]
        for i, pos in enumerate(self.agent_positions):
            circle = patches.Circle(pos, 0.3, color=colors[i % len(colors)], alpha=0.7)
            ax.add_patch(circle)
            ax.text(
                pos[0], pos[1], f"A{i}", ha="center", va="center", fontweight="bold"
            )

        # Draw goals
        for i, pos in enumerate(self.goal_positions):
            square = patches.Rectangle(
                (pos[0] - 0.4, pos[1] - 0.4), 0.8, 0.8, color="gold", alpha=0.7
            )
            ax.add_patch(square)
            ax.text(
                pos[0], pos[1], f"G{i}", ha="center", va="center", fontweight="bold"
            )

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect("equal")
        ax.set_title("Multi-Agent Cooperation Environment")

        plt.show()


class ContinuousControlEnv:
    """Continuous control environment with complex dynamics."""

    def __init__(self, state_dim=6, action_dim=2, max_steps=200):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(state_dim,), dtype=np.float32
        )

        # Physical parameters
        self.mass = 1.0
        self.damping = 0.1
        self.spring_constant = 1.0

        self.reset()

    def reset(self):
        """Reset environment."""
        self.state = np.random.uniform(-2, 2, self.state_dim)
        self.step_count = 0

        # Set target
        self.target = np.random.uniform(-3, 3, 2)  # Only first 2 dimensions matter

        return self.state.copy()

    def step(self, action):
        """Take step with continuous action."""
        self.step_count += 1

        # Apply action with saturation
        action = np.clip(action, -1, 1)

        # Complex dynamics simulation
        dt = 0.1

        # Position and velocity
        pos = self.state[: self.state_dim // 2]
        vel = self.state[self.state_dim // 2 :]

        # Forces
        spring_force = -self.spring_constant * pos
        damping_force = -self.damping * vel
        control_force = action * 2.0  # Scale control force

        # Acceleration
        accel = (spring_force + damping_force + control_force) / self.mass

        # Update state
        new_vel = vel + accel * dt
        new_pos = pos + new_vel * dt

        self.state = np.concatenate([new_pos, new_pos])  # Simple state representation

        # Reward calculation
        distance_to_target = np.linalg.norm(pos - self.target)
        reward = -distance_to_target - 0.1 * np.linalg.norm(action)

        # Bonus for reaching target
        if distance_to_target < 0.5:
            reward += 10

        # Done condition
        done = self.step_count >= self.max_steps or distance_to_target < 0.1

        return self.state.copy(), reward, done, {"target": self.target}

    def render(self):
        """Render environment."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw agent position
        pos = self.state[:2]
        circle = patches.Circle(pos, 0.2, color="blue", alpha=0.7)
        ax.add_patch(circle)

        # Draw target
        target_circle = patches.Circle(self.target, 0.3, color="red", alpha=0.7)
        ax.add_patch(target_circle)

        # Draw velocity vector
        vel = self.state[2:4]
        ax.arrow(
            pos[0],
            pos[1],
            vel[0],
            vel[1],
            head_width=0.1,
            head_length=0.1,
            fc="green",
            ec="green",
            alpha=0.7,
        )

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_title("Continuous Control Environment")
        ax.grid(True, alpha=0.3)

        plt.show()


class HierarchicalTaskEnv:
    """Hierarchical task environment with multiple levels of subgoals."""

    def __init__(self, num_levels=3, tasks_per_level=2):
        self.num_levels = num_levels
        self.tasks_per_level = tasks_per_level

        # Task hierarchy
        self.task_hierarchy = self._create_task_hierarchy()
        self.current_level = 0
        self.current_task = 0

        # State and action spaces
        self.state_dim = num_levels * tasks_per_level + 2  # Task states + position
        self.action_dim = 4  # Move in 4 directions

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.state_dim,), dtype=np.float32
        )

        self.position = np.array([0.0, 0.0])
        self.reset()

    def _create_task_hierarchy(self):
        """Create hierarchical task structure."""
        hierarchy = {}

        for level in range(self.num_levels):
            hierarchy[level] = []
            for task in range(self.tasks_per_level):
                hierarchy[level].append(
                    {
                        "name": f"Level{level}_Task{task}",
                        "position": np.random.uniform(-2, 2, 2),
                        "completed": False,
                        "reward": 10 * (level + 1),  # Higher level = more reward
                    }
                )

        return hierarchy

    def reset(self):
        """Reset environment."""
        # Reset all tasks
        for level in range(self.num_levels):
            for task in range(self.tasks_per_level):
                self.task_hierarchy[level][task]["completed"] = False

        self.current_level = 0
        self.current_task = 0
        self.position = np.array([0.0, 0.0])

        return self._get_observation()

    def step(self, action):
        """Take step in hierarchical environment."""
        # Move agent
        if action == 0:  # Up
            self.position[1] += 0.1
        elif action == 1:  # Down
            self.position[1] -= 0.1
        elif action == 2:  # Left
            self.position[0] -= 0.1
        elif action == 3:  # Right
            self.position[0] += 0.1

        # Clamp position
        self.position = np.clip(self.position, -3, 3)

        reward = 0
        done = False

        # Check if current task is completed
        current_task_info = self.task_hierarchy[self.current_level][self.current_task]
        distance_to_task = np.linalg.norm(self.position - current_task_info["position"])

        if distance_to_task < 0.2 and not current_task_info["completed"]:
            # Task completed
            current_task_info["completed"] = True
            reward += current_task_info["reward"]

            # Move to next task
            self.current_task += 1

            # Check if level completed
            if self.current_task >= self.tasks_per_level:
                self.current_level += 1
                self.current_task = 0

                # Check if all levels completed
                if self.current_level >= self.num_levels:
                    done = True
                    reward += 100  # Bonus for completing all levels

        # Small penalty for each step
        reward -= 0.01

        return (
            self._get_observation(),
            reward,
            done,
            {
                "current_level": self.current_level,
                "current_task": self.current_task,
                "task_completed": current_task_info["completed"],
            },
        )

    def _get_observation(self):
        """Get observation."""
        obs = np.zeros(self.state_dim)

        # Task completion status
        idx = 0
        for level in range(self.num_levels):
            for task in range(self.tasks_per_level):
                obs[idx] = 1.0 if self.task_hierarchy[level][task]["completed"] else 0.0
                idx += 1

        # Current position
        obs[idx] = (self.position[0] + 3) / 6  # Normalize to [0, 1]
        obs[idx + 1] = (self.position[1] + 3) / 6

        return obs.astype(np.float32)

    def render(self):
        """Render hierarchical environment."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw tasks
        colors = ["red", "orange", "yellow", "green", "blue", "purple"]
        for level in range(self.num_levels):
            for task in range(self.tasks_per_level):
                task_info = self.task_hierarchy[level][task]
                pos = task_info["position"]

                if task_info["completed"]:
                    # Completed task
                    circle = patches.Circle(pos, 0.3, color="lightgreen", alpha=0.7)
                else:
                    # Pending task
                    circle = patches.Circle(
                        pos, 0.3, color=colors[level % len(colors)], alpha=0.7
                    )

                ax.add_patch(circle)
                ax.text(
                    pos[0],
                    pos[1],
                    f"L{level}T{task}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        # Draw agent
        agent_circle = patches.Circle(self.position, 0.15, color="black", alpha=0.8)
        ax.add_patch(agent_circle)

        # Draw current task target
        if (
            self.current_level < self.num_levels
            and self.current_task < self.tasks_per_level
        ):
            current_task_info = self.task_hierarchy[self.current_level][
                self.current_task
            ]
            if not current_task_info["completed"]:
                target_pos = current_task_info["position"]
                target_circle = patches.Circle(
                    target_pos, 0.4, color="red", fill=False, linewidth=3
                )
                ax.add_patch(target_circle)

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_title(
            f"Hierarchical Task Environment - Level {self.current_level}, Task {self.current_task}"
        )
        ax.grid(True, alpha=0.3)

        plt.show()


class DynamicObstacleEnv:
    """Environment with dynamic obstacles and changing goals."""

    def __init__(self, grid_size=15, num_obstacles=5, num_goals=3):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.num_goals = num_goals

        self.action_space = spaces.Discrete(4)  # 4 directions
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size * grid_size + 4,), dtype=np.float32
        )

        # Agent and goal positions
        self.agent_pos = np.array([0, 0])
        self.goal_positions = []

        # Dynamic obstacles
        self.obstacle_positions = []
        self.obstacle_velocities = []

        # Goal change frequency
        self.goal_change_freq = 50
        self.step_count = 0

        self.reset()

    def reset(self):
        """Reset environment."""
        self.step_count = 0

        # Place agent
        self.agent_pos = np.array([0, 0])

        # Place goals
        self.goal_positions = []
        for _ in range(self.num_goals):
            pos = np.array(
                [
                    random.randint(1, self.grid_size - 1),
                    random.randint(1, self.grid_size - 1),
                ]
            )
            self.goal_positions.append(pos)

        # Place obstacles
        self.obstacle_positions = []
        self.obstacle_velocities = []
        for _ in range(self.num_obstacles):
            pos = np.array(
                [
                    random.randint(1, self.grid_size - 1),
                    random.randint(1, self.grid_size - 1),
                ]
            )
            vel = np.array([random.choice([-1, 0, 1]), random.choice([-1, 0, 1])])
            self.obstacle_positions.append(pos)
            self.obstacle_velocities.append(vel)

        return self._get_observation()

    def step(self, action):
        """Take step in dynamic environment."""
        self.step_count += 1

        # Move agent
        if action == 0:  # Up
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 1:  # Down
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 2:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # Right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)

        # Move obstacles
        for i in range(self.num_obstacles):
            self.obstacle_positions[i] += self.obstacle_velocities[i]

            # Bounce off walls
            if (
                self.obstacle_positions[i][0] <= 0
                or self.obstacle_positions[i][0] >= self.grid_size - 1
            ):
                self.obstacle_velocities[i][0] *= -1
            if (
                self.obstacle_positions[i][1] <= 0
                or self.obstacle_positions[i][1] >= self.grid_size - 1
            ):
                self.obstacle_velocities[i][1] *= -1

            # Clamp position
            self.obstacle_positions[i] = np.clip(
                self.obstacle_positions[i], 0, self.grid_size - 1
            )

        # Change goals periodically
        if self.step_count % self.goal_change_freq == 0:
            for i in range(self.num_goals):
                self.goal_positions[i] = np.array(
                    [
                        random.randint(1, self.grid_size - 1),
                        random.randint(1, self.grid_size - 1),
                    ]
                )

        # Check collisions
        reward = -0.01  # Small negative reward for each step

        # Check obstacle collision
        for obstacle_pos in self.obstacle_positions:
            if np.allclose(self.agent_pos, obstacle_pos):
                reward -= 10  # Penalty for collision

        # Check goal achievement
        for i, goal_pos in enumerate(self.goal_positions):
            if np.allclose(self.agent_pos, goal_pos):
                reward += 10 * (i + 1)  # Different rewards for different goals

        # Check if all goals achieved
        all_goals_achieved = True
        for goal_pos in self.goal_positions:
            if not np.allclose(self.agent_pos, goal_pos):
                all_goals_achieved = False
                break

        done = all_goals_achieved or self.step_count >= 500

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """Get observation."""
        # Create grid representation
        grid = np.zeros((self.grid_size, self.grid_size))

        # Mark agent position
        grid[int(self.agent_pos[1]), int(self.agent_pos[0])] = 1

        # Mark goal positions
        for i, goal_pos in enumerate(self.goal_positions):
            grid[int(goal_pos[1]), int(goal_pos[0])] = 0.5

        # Mark obstacle positions
        for obstacle_pos in self.obstacle_positions:
            grid[int(obstacle_pos[1]), int(obstacle_pos[0])] = -1

        # Flatten grid and add agent position
        obs = np.concatenate([grid.flatten(), self.agent_pos / self.grid_size])

        return obs.astype(np.float32)

    def render(self):
        """Render dynamic environment."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color="black", linewidth=0.5)
            ax.axvline(i - 0.5, color="black", linewidth=0.5)

        # Draw agent
        agent_circle = patches.Circle(self.agent_pos, 0.3, color="blue", alpha=0.8)
        ax.add_patch(agent_circle)

        # Draw goals
        for i, goal_pos in enumerate(self.goal_positions):
            goal_circle = patches.Circle(goal_pos, 0.3, color="green", alpha=0.7)
            ax.add_patch(goal_circle)
            ax.text(
                goal_pos[0],
                goal_pos[1],
                f"G{i}",
                ha="center",
                va="center",
                fontweight="bold",
            )

        # Draw obstacles
        for i, obstacle_pos in enumerate(self.obstacle_positions):
            obstacle_rect = patches.Rectangle(
                (obstacle_pos[0] - 0.3, obstacle_pos[1] - 0.3),
                0.6,
                0.6,
                color="red",
                alpha=0.7,
            )
            ax.add_patch(obstacle_rect)

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect("equal")
        ax.set_title("Dynamic Obstacle Environment")

        plt.show()


