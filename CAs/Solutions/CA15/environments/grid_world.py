"""
Reinforcement Learning Environments

This module contains custom environments for testing RL algorithms,
including grid worlds and other control tasks.
"""

import numpy as np


class SimpleGridWorld:
    """Simple grid world for model-based vs model-free comparison."""

    def __init__(self, size=8, num_goals=1):
        self.size = size
        self.num_goals = num_goals
        self.action_space_size = 4  # up, down, left, right
        self.state_dim = size * size
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.agent_pos = [0, 0]

        # Place goal randomly
        self.goal_pos = [
            np.random.randint(self.size // 2, self.size),
            np.random.randint(self.size // 2, self.size),
        ]

        # Ensure agent and goal are different
        while self.agent_pos == self.goal_pos:
            self.goal_pos = [
                np.random.randint(1, self.size),
                np.random.randint(1, self.size),
            ]

        self.steps = 0
        self.max_steps = self.size * 4

        return self._get_state()

    def _get_state(self):
        """Convert position to state representation."""
        state = np.zeros(self.state_dim)
        agent_idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        goal_idx = self.goal_pos[0] * self.size + self.goal_pos[1]

        state[agent_idx] = 1.0  # Agent position
        state[goal_idx] = 0.5  # Goal position

        return state

    def step(self, action):
        """Execute action and return next state, reward, done."""
        # Actions: 0=up, 1=down, 2=left, 3=right
        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        if action < len(moves):
            new_pos = [
                self.agent_pos[0] + moves[action][0],
                self.agent_pos[1] + moves[action][1],
            ]

            # Clip to boundaries
            new_pos[0] = max(0, min(self.size - 1, new_pos[0]))
            new_pos[1] = max(0, min(self.size - 1, new_pos[1]))

            self.agent_pos = new_pos

        self.steps += 1

        # Calculate reward
        distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(
            self.agent_pos[1] - self.goal_pos[1]
        )

        if distance == 0:
            reward = 100.0  # Goal reached
            done = True
        else:
            reward = -1.0 - 0.1 * distance  # Step penalty + distance penalty
            done = False

        # Episode timeout
        if self.steps >= self.max_steps:
            done = True
            if distance > 0:
                reward -= 50.0  # Timeout penalty

        info = {"distance": distance, "steps": self.steps}

        return self._get_state(), reward, done, info
