"""
Robust Environment for Robust Reinforcement Learning

This module provides environments with configurable uncertainty for testing robust RL algorithms.
"""

import numpy as np


class RobustEnvironment:
    """Environment with configurable uncertainty for robust RL."""

    def __init__(self, base_size=6, uncertainty_level=0.1, dynamic_obstacles=True):
        self.base_size = base_size
        self.uncertainty_level = uncertainty_level
        self.dynamic_obstacles = dynamic_obstacles

        # Environment parameters (can be randomized)
        self.current_size = base_size
        self.noise_std = 0.0
        self.action_failure_prob = 0.0
        self.reward_noise_std = 0.0

        # State
        self.reset()

        self.action_space = 4  # up, down, left, right
        self.max_episode_steps = 100

    def randomize_parameters(self):
        """Apply domain randomization to environment parameters."""
        # Randomize grid size
        size_variation = max(1, int(self.base_size * self.uncertainty_level))
        self.current_size = np.random.randint(
            max(3, self.base_size - size_variation), self.base_size + size_variation + 1
        )

        # Randomize noise levels
        self.noise_std = np.random.uniform(0, self.uncertainty_level)
        self.action_failure_prob = np.random.uniform(0, self.uncertainty_level)
        self.reward_noise_std = np.random.uniform(0, self.uncertainty_level * 5)

        # Randomize obstacle positions
        if self.dynamic_obstacles:
            num_obstacles = np.random.randint(0, max(1, self.current_size // 2))
            self.obstacles = []
            for _ in range(num_obstacles):
                obs_pos = [
                    np.random.randint(1, self.current_size - 1),
                    np.random.randint(1, self.current_size - 1),
                ]
                if obs_pos not in self.obstacles:
                    self.obstacles.append(obs_pos)

    def reset(self):
        """Reset environment with potential randomization."""
        # Apply domain randomization
        self.randomize_parameters()

        # Reset agent position
        self.agent_pos = [0, 0]
        self.goal_pos = [self.current_size - 1, self.current_size - 1]
        self.current_step = 0

        # Initialize obstacles if not already set
        if not hasattr(self, "obstacles"):
            self.obstacles = []

        return self.get_observation()

    def get_observation(self):
        """Get observation with potential noise."""
        obs = np.array(
            [
                self.agent_pos[0] / self.current_size,
                self.agent_pos[1] / self.current_size,
                (self.goal_pos[0] - self.agent_pos[0]) / self.current_size,
                (self.goal_pos[1] - self.agent_pos[1]) / self.current_size,
                self.current_size / 10.0,  # Environment size as feature
                len(self.obstacles) / 10.0,  # Number of obstacles
            ],
            dtype=np.float32,
        )

        # Add observation noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, obs.shape)
            obs += noise

        return obs

    def step(self, action):
        """Execute action with potential failures and noise."""
        self.current_step += 1

        # Action failure (agent doesn't move)
        if np.random.random() < self.action_failure_prob:
            action = 4  # Stay in place

        prev_pos = self.agent_pos.copy()

        # Execute action
        if action == 0 and self.agent_pos[1] < self.current_size - 1:  # up
            self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0:  # down
            self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0:  # left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.current_size - 1:  # right
            self.agent_pos[0] += 1

        # Check obstacle collision
        if self.agent_pos in self.obstacles:
            self.agent_pos = prev_pos  # Revert move
            reward = -5.0  # Collision penalty
        else:
            # Regular reward
            done = self.agent_pos == self.goal_pos
            if done:
                reward = 10.0
            else:
                # Distance-based reward
                dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(
                    self.agent_pos[1] - self.goal_pos[1]
                )
                reward = -0.1 - 0.01 * dist

        # Add reward noise
        if self.reward_noise_std > 0:
            reward += np.random.normal(0, self.reward_noise_std)

        done = (self.agent_pos == self.goal_pos) or (
            self.current_step >= self.max_episode_steps
        )

        info = {
            "environment_size": self.current_size,
            "noise_level": self.noise_std,
            "action_failure_prob": self.action_failure_prob,
            "obstacles": len(self.obstacles),
        }

        return self.get_observation(), reward, done, info
