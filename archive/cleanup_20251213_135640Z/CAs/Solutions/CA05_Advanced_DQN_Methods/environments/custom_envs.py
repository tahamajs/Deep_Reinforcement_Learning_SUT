"""
Custom environments for CA5 Advanced DQN Methods
"""

import gym
import numpy as np
from gym import spaces
from typing import Tuple, Any, Dict


class GridWorldEnv(gym.Env):
    """
    Simple Grid World environment for testing DQN algorithms
    """

    def __init__(self, size: int = 5):
        super().__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(size, size), dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.state = np.zeros((self.size, self.size), dtype=np.float32)
        self.state[self.agent_pos[0], self.agent_pos[1]] = 1.0
        self.state[self.goal_pos[0], self.goal_pos[1]] = 0.5
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action and return next state, reward, done, info"""
        # Action mapping: 0=Up, 1=Down, 2=Left, 3=Right
        action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        # Calculate new position
        dx, dy = action_map[action]
        new_pos = [
            max(0, min(self.size - 1, self.agent_pos[0] + dx)),
            max(0, min(self.size - 1, self.agent_pos[1] + dy)),
        ]

        # Update state
        self.state[self.agent_pos[0], self.agent_pos[1]] = 0.0
        self.agent_pos = new_pos
        self.state[self.agent_pos[0], self.agent_pos[1]] = 1.0

        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        else:
            reward = -0.1  # Small negative reward for each step
            done = False

        info = {"agent_pos": self.agent_pos.copy(), "goal_pos": self.goal_pos.copy()}

        return self.state, reward, done, info

    def render(self, mode: str = "human") -> Any:
        """Render the environment"""
        if mode == "human":
            print(f"Agent at: {self.agent_pos}, Goal at: {self.goal_pos}")
            print(self.state)
        return self.state


class MountainCarContinuousEnv(gym.Env):
    """
    Continuous Mountain Car environment wrapper
    """

    def __init__(self):
        super().__init__()
        self.env = gym.make("MountainCarContinuous-v0")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self.env.step(action)

    def render(self, mode: str = "human") -> Any:
        return self.env.render(mode)

    def close(self):
        self.env.close()


class LunarLanderEnv(gym.Env):
    """
    Lunar Lander environment wrapper with custom reward shaping
    """

    def __init__(self, reward_shaping: bool = True):
        super().__init__()
        self.env = gym.make("LunarLander-v2")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_shaping = reward_shaping

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        state, reward, done, info = self.env.step(action)

        if self.reward_shaping:
            # Add reward shaping based on state
            x, y, vx, vy, angle, vang, leg1, leg2 = state

            # Reward for being close to landing pad
            distance_reward = -abs(x) * 0.1

            # Penalty for high speed
            speed_penalty = -abs(vx) * 0.1 - abs(vy) * 0.1

            # Reward for good angle
            angle_reward = -abs(angle) * 0.1

            reward += distance_reward + speed_penalty + angle_reward

        return state, reward, done, info

    def render(self, mode: str = "human") -> Any:
        return self.env.render(mode)

    def close(self):
        self.env.close()


def make_env(env_name: str, **kwargs) -> gym.Env:
    """
    Factory function to create environments
    """
    if env_name == "GridWorld":
        return GridWorldEnv(**kwargs)
    elif env_name == "MountainCarContinuous":
        return MountainCarContinuousEnv()
    elif env_name == "LunarLander":
        return LunarLanderEnv(**kwargs)
    else:
        return gym.make(env_name)


if __name__ == "__main__":
    # Test environments
    print("Testing GridWorld environment...")
    env = GridWorldEnv(size=5)
    state = env.reset()
    print(f"Initial state shape: {state.shape}")

    for i in range(10):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.2f}, Done={done}")
        if done:
            break

    env.close()
    print("GridWorld test completed!")


