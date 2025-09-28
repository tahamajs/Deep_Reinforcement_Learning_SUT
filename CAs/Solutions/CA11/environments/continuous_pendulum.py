"""
Continuous Pendulum Environment
"""

import numpy as np
import math


class ContinuousPendulum:
    """Continuous pendulum environment"""

    def __init__(self):
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # torque
        self.max_torque = 2.0
        self.max_speed = 8.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0

        self.state = None
        self.step_count = 0
        self.max_steps = 200

    def reset(self):
        theta = np.random.uniform(-np.pi, np.pi)
        theta_dot = np.random.uniform(-1, 1)
        self.state = np.array([np.cos(theta), np.sin(theta), theta_dot])
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()

        action = np.clip(action, -self.max_torque, self.max_torque)

        cos_theta, sin_theta, theta_dot = self.state
        theta = np.arctan2(sin_theta, cos_theta)

        theta_dot_dot = (
            3 * self.g / (2 * self.l) * np.sin(theta)
            + 3 / (self.m * self.l**2) * action
        )
        theta_dot = theta_dot + theta_dot_dot * self.dt
        theta_dot = np.clip(theta_dot, -self.max_speed, self.max_speed)
        theta = theta + theta_dot * self.dt

        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([np.cos(theta), np.sin(theta), theta_dot])

        reward = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * action**2)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self.state.copy(), reward, done
