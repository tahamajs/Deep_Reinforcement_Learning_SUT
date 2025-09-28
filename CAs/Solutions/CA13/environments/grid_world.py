import numpy as np


class SimpleGridWorld:
    """Simple gridworld environment for reinforcement learning demonstrations."""

    def __init__(self, size=5):
        """Initialize the gridworld environment.

        Args:
            size (int): Size of the grid (size x size)
        """
        self.size = size
        self.num_actions = 4  # up, down, left, right
        self.goal = [size - 1, size - 1]  # Bottom-right corner
        self.reset()

    def reset(self):
        """Reset the environment to initial state.

        Returns:
            np.ndarray: Initial state [row, col]
        """
        self.state = [0, 0]  # Start at top-left corner
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        """Take a step in the environment.

        Args:
            action (int): Action to take (0=up, 1=down, 2=left, 3=right)

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # Apply action with boundary checking
        new_row = max(0, min(self.size - 1, self.state[0] + moves[action][0]))
        new_col = max(0, min(self.size - 1, self.state[1] + moves[action][1]))

        self.state = [new_row, new_col]

        # Reward structure
        if self.state == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -0.1  # Small negative reward for each step
            done = False

        return np.array(self.state, dtype=np.float32), reward, done, {}
