"""
Hierarchical Reinforcement Learning Environments

This module contains custom environments designed for testing hierarchical RL algorithms,
including multi-goal navigation tasks and environments that benefit from temporal abstraction.
"""

import numpy as np


class HierarchicalRLEnvironment:
    """Custom environment for testing hierarchical RL algorithms."""

    def __init__(self, size=10, num_goals=3):
        self.size = size
        self.num_goals = num_goals
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.agent_pos = np.array([0, 0])
        self.goals = []

        for _ in range(self.num_goals):
            goal_pos = np.random.randint(0, self.size, size=2)
            while np.array_equal(goal_pos, self.agent_pos):
                goal_pos = np.random.randint(0, self.size, size=2)
            self.goals.append(goal_pos)

        self.current_goal_idx = 0
        self.steps = 0
        self.max_steps = self.size * 4

        return self.get_state()

    def get_state(self):
        """Get current state representation."""
        state = np.zeros((self.size, self.size))
        state[self.agent_pos[0], self.agent_pos[1]] = 1.0  # Agent position

        for i, goal in enumerate(self.goals):
            if i == self.current_goal_idx:
                state[goal[0], goal[1]] = 0.5  # Current goal
            else:
                state[goal[0], goal[1]] = 0.3  # Other goals

        return state.flatten()

    def step(self, action):
        """Execute action and return next state, reward, done."""
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        if action < len(moves):
            new_pos = self.agent_pos + np.array(moves[action])
            new_pos = np.clip(new_pos, 0, self.size - 1)
            self.agent_pos = new_pos

        self.steps += 1

        reward = 0
        done = False

        current_goal = self.goals[self.current_goal_idx]
        if np.array_equal(self.agent_pos, current_goal):
            reward = 10.0  # Goal reached
            self.current_goal_idx += 1

            if self.current_goal_idx >= self.num_goals:
                done = True  # All goals reached
                reward += 50.0  # Bonus for completing all goals
        else:
            distance = np.linalg.norm(self.agent_pos - current_goal)
            reward = -0.1 * distance

        if self.steps >= self.max_steps:
            done = True
            reward -= 10.0  # Penalty for timeout

        info = {
            "goals_completed": self.current_goal_idx,
            "current_goal": current_goal,
            "agent_pos": self.agent_pos.copy(),
        }

        return self.get_state(), reward, done, info
