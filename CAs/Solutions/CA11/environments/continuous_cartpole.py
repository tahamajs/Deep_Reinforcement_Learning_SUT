"""
Continuous CartPole Environment
"""

import numpy as np
import math


class ContinuousCartPole:
    """Continuous version of CartPole for world model testing"""

    def __init__(self):
        self.state_dim = 4
        self.action_dim = 1
        self.max_steps = 200
        self.current_step = 0

        # Physics parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # State bounds
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * math.pi / 360

        self.state = None
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.state = np.random.uniform(-0.05, 0.05, size=(4,))
        self.current_step = 0
        return self.state.copy()

    def step(self, action):
        """Take environment step"""
        if isinstance(action, np.ndarray):
            action = action.item()

        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        force = action * self.force_mag

        x, x_dot, theta, theta_dot = self.state

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Physics calculations
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Update state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.current_step += 1

        # Calculate reward and done
        done = (
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians or
            self.current_step >= self.max_steps
        )

        reward = 1.0 if not done else 0.0

        return self.state.copy(), reward, done

    def sample_action(self):
        """Sample random action"""
        return np.random.uniform(-1.0, 1.0)