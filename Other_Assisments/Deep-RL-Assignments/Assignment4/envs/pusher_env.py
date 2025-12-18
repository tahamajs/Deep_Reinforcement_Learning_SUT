"""2D Pushing Environment for MPC assignment."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Pushing2DEnv(gym.Env):
    """2D Pushing Environment."""

    def __init__(self, max_steps=40, noise_level=0.0):
        self.max_steps = max_steps
        self.noise_level = noise_level

        # State: [pusher_x, pusher_y, box_x, box_y, goal_x, goal_y, pusher_vx, pusher_vy]
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, -5, -5, -5, -5, -10, -10]),
            high=np.array([5, 5, 5, 5, 5, 5, 10, 10]),
            dtype=np.float32
        )

        # Action: [force_x, force_y] applied to pusher
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        self.dt = 0.1
        self.pusher_mass = 1.0
        self.box_mass = 2.0
        self.pusher_radius = 0.2
        self.box_size = 0.4
        self.contact_distance = self.pusher_radius + self.box_size / 2

        self.reset()

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        # Random initial positions
        self.pusher_pos = np.random.uniform(-2, 2, 2)
        self.box_pos = np.random.uniform(-2, 2, 2)
        self.goal_pos = np.random.uniform(-2, 2, 2)

        # Ensure minimum distance
        while np.linalg.norm(self.pusher_pos - self.box_pos) < 1.0:
            self.box_pos = np.random.uniform(-2, 2, 2)
        while np.linalg.norm(self.box_pos - self.goal_pos) < 1.0:
            self.goal_pos = np.random.uniform(-2, 2, 2)

        self.pusher_vel = np.zeros(2)
        self.steps = 0

        state = np.concatenate([
            self.pusher_pos, self.box_pos, self.goal_pos, self.pusher_vel
        ])
        return state.astype(np.float32), {}

    def step(self, action):
        """Step the environment."""
        # Apply force to pusher
        force = np.clip(action, -1, 1)
        self.pusher_vel += (force / self.pusher_mass) * self.dt

        # Add friction
        self.pusher_vel *= 0.9

        # Update pusher position
        self.pusher_pos += self.pusher_vel * self.dt

        # Check collision with box
        distance = np.linalg.norm(self.pusher_pos - self.box_pos)
        if distance < self.contact_distance:
            # Collision: transfer momentum to box
            direction = (self.box_pos - self.pusher_pos) / (distance + 1e-6)
            impulse = 0.5 * np.dot(self.pusher_vel, direction) * direction
            self.pusher_vel -= impulse
            self.box_pos += (impulse / self.box_mass) * self.dt

        # Keep positions in bounds
        self.pusher_pos = np.clip(self.pusher_pos, -4.5, 4.5)
        self.box_pos = np.clip(self.box_pos, -4.5, 4.5)

        self.steps += 1

        # Check termination
        done = self.steps >= self.max_steps
        distance_to_goal = np.linalg.norm(self.box_pos - self.goal_pos)
        success = distance_to_goal < 0.5

        if success:
            reward = 0.0  # Success reward
        else:
            reward = -distance_to_goal  # Distance penalty

        state = np.concatenate([
            self.pusher_pos, self.box_pos, self.goal_pos, self.pusher_vel
        ])

        return state.astype(np.float32), reward, done, False, {}

    def get_nxt_state(self, state, action):
        """Get next state without changing environment state."""
        # Unpack state
        pusher_pos = state[:2]
        box_pos = state[2:4]
        goal_pos = state[4:6]
        pusher_vel = state[6:8]

        # Apply force to pusher
        force = np.clip(action, -1, 1)
        new_pusher_vel = pusher_vel + (force / self.pusher_mass) * self.dt

        # Add friction
        new_pusher_vel *= 0.9

        # Update pusher position
        new_pusher_pos = pusher_pos + new_pusher_vel * self.dt

        # Check collision with box
        distance = np.linalg.norm(new_pusher_pos - box_pos)
        new_box_pos = box_pos.copy()
        if distance < self.contact_distance:
            # Collision: transfer momentum to box
            direction = (box_pos - new_pusher_pos) / (distance + 1e-6)
            impulse = 0.5 * np.dot(new_pusher_vel, direction) * direction
            new_pusher_vel = new_pusher_vel - impulse
            new_box_pos = box_pos + (impulse / self.box_mass) * self.dt

        # Keep positions in bounds
        new_pusher_pos = np.clip(new_pusher_pos, -4.5, 4.5)
        new_box_pos = np.clip(new_box_pos, -4.5, 4.5)

        next_state = np.concatenate([
            new_pusher_pos, new_box_pos, goal_pos, new_pusher_vel
        ])

        return next_state.astype(np.float32)

    def render(self, mode='human'):
        """Render the environment."""
        # Simple text-based rendering for now
        print(f"Pusher: {self.pusher_pos}, Box: {self.box_pos}, Goal: {self.goal_pos}")
        return None


class Pushing2DNoisyControlEnv(Pushing2DEnv):
    """2D Pushing Environment with noisy control."""

    def __init__(self, max_steps=40, noise_level=0.1):
        super().__init__(max_steps=max_steps, noise_level=noise_level)

    def step(self, action):
        """Step with noisy control."""
        noisy_action = action + np.random.normal(0, self.noise_level, size=action.shape)
        return super().step(noisy_action)

    def get_nxt_state(self, state, action):
        """Get next state with noisy control."""
        noisy_action = action + np.random.normal(0, self.noise_level, size=action.shape)
        return super().get_nxt_state(state, noisy_action)