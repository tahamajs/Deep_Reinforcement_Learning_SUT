"""
Safe Environment for Safe Reinforcement Learning

This module provides environments with safety constraints for testing safe RL algorithms.
"""

import numpy as np


class SafeEnvironment:
    """Environment with safety constraints for Safe RL demonstration."""

    def __init__(self, size=6, hazard_positions=None, constraint_threshold=0.1):
        self.size = size
        self.state = [0, 0]
        self.goal = [size - 1, size - 1]
        self.constraint_threshold = constraint_threshold

        # Define hazardous areas
        if hazard_positions is None:
            self.hazards = [[2, 2], [3, 1], [1, 3], [4, 3]]
        else:
            self.hazards = hazard_positions

        self.action_space = 4  # up, down, left, right
        self.max_episode_steps = 50
        self.current_step = 0

        # Safety statistics
        self.constraint_violations = 0
        self.total_constraint_cost = 0

    def reset(self):
        """Reset environment to initial state."""
        self.state = [0, 0]
        self.current_step = 0
        self.constraint_violations = 0
        self.total_constraint_cost = 0
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        """Take action in environment with safety constraints."""
        self.current_step += 1

        # Execute action
        prev_state = self.state.copy()
        if action == 0 and self.state[1] < self.size - 1:  # up
            self.state[1] += 1
        elif action == 1 and self.state[1] > 0:  # down
            self.state[1] -= 1
        elif action == 2 and self.state[0] > 0:  # left
            self.state[0] -= 1
        elif action == 3 and self.state[0] < self.size - 1:  # right
            self.state[0] += 1

        # Compute reward
        done = self.state == self.goal
        reward = 10.0 if done else -0.1

        # Compute constraint cost (safety violations)
        constraint_cost = self._compute_constraint_cost(self.state)

        # Check if episode should terminate
        episode_done = done or self.current_step >= self.max_episode_steps

        info = {
            "constraint_cost": constraint_cost,
            "constraint_violation": constraint_cost > 0,
            "total_violations": self.constraint_violations,
            "position": self.state.copy(),
        }

        return np.array(self.state, dtype=np.float32), reward, episode_done, info

    def _compute_constraint_cost(self, state):
        """Compute constraint violation cost."""
        cost = 0.0

        # Hazard penalty
        if state in self.hazards:
            cost += 1.0  # High cost for being in hazardous areas
            self.constraint_violations += 1

        # Boundary penalty (soft constraints)
        if (
            state[0] == 0
            or state[0] == self.size - 1
            or state[1] == 0
            or state[1] == self.size - 1
        ):
            cost += 0.1  # Small cost for being near boundaries

        self.total_constraint_cost += cost
        return cost

    def is_safe_state(self, state):
        """Check if state is safe (no constraint violations)."""
        return state not in self.hazards

    def get_safe_actions(self, state):
        """Get list of safe actions from current state."""
        safe_actions = []
        for action in range(self.action_space):
            next_state = state.copy()
            if action == 0 and state[1] < self.size - 1:
                next_state[1] += 1
            elif action == 1 and state[1] > 0:
                next_state[1] -= 1
            elif action == 2 and state[0] > 0:
                next_state[0] -= 1
            elif action == 3 and state[0] < self.size - 1:
                next_state[0] += 1

            if self.is_safe_state(next_state):
                safe_actions.append(action)

        return safe_actions if safe_actions else list(range(self.action_space))
