import numpy as np


class SimpleGridWorld:
    """Simple grid world environment for offline RL demonstrations."""

    def __init__(self, size=5):
        self.size = size
        self.state = [0, 0]
        self.goal = [size - 1, size - 1]
        self.action_space = 4  # up, down, left, right

    def reset(self):
        self.state = [0, 0]
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        if action == 0 and self.state[1] < self.size - 1:
            self.state[1] += 1
        elif action == 1 and self.state[1] > 0:
            self.state[1] -= 1
        elif action == 2 and self.state[0] > 0:
            self.state[0] -= 1
        elif action == 3 and self.state[0] < self.size - 1:
            self.state[0] += 1

        done = self.state == self.goal
        reward = 1.0 if done else -0.1

        return np.array(self.state, dtype=np.float32), reward, done, {}
