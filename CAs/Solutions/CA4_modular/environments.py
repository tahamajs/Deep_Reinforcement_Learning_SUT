"""
Environment utilities for Policy Gradient Methods
CA4: Policy Gradient Methods and Neural Networks in RL
"""

import gym
import numpy as np
import torch
from typing import Optional, Tuple, Any


class EnvironmentWrapper:
    """Wrapper for reinforcement learning environments"""

    def __init__(self, env_name: str = "CartPole-v1"):
        """Initialize environment wrapper

        Args:
            env_name: Name of the gym environment
        """
        try:
            self.env = gym.make(env_name)
            self.env_name = env_name
            self.state_size = self.env.observation_space.shape[0]
            self.action_size = (
                self.env.action_space.n if hasattr(self.env.action_space, "n") else None
            )
            self.is_continuous = not hasattr(self.env.action_space, "n")
        except Exception as e:
            print(f"Environment {env_name} not available: {e}")
            self.env = None
            self.state_size = 4  # Default for CartPole-like
            self.action_size = 2
            self.is_continuous = False

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        if self.env is not None:
            state, _ = self.env.reset()
            return np.array(state)
        return np.random.randn(self.state_size)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take action in environment"""
        if self.env is not None:
            if isinstance(action, torch.Tensor):
                action = action.item()
            elif isinstance(action, np.ndarray):
                action = action.item() if action.ndim == 0 else action[0]

            next_state, reward, done, truncated, info = self.env.step(action)
            return np.array(next_state), reward, done, truncated, info

        # Mock environment for demonstration
        next_state = np.random.randn(self.state_size)
        reward = 1.0 if np.random.rand() > 0.5 else -1.0
        done = np.random.rand() > 0.95
        truncated = False
        info = {}
        return next_state, reward, done, truncated, info

    def close(self):
        """Close environment"""
        if self.env is not None:
            self.env.close()

    def render(self):
        """Render environment"""
        if self.env is not None:
            return self.env.render()
        return None


class PolicyDemoEnvironment:
    """Simple environment for policy demonstration"""

    def __init__(self, n_states: int = 4, n_actions: int = 2):
        """Initialize demo environment

        Args:
            n_states: Number of states
            n_actions: Number of actions
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.current_state = 0

        # Define transition dynamics
        self.transitions = np.random.rand(n_states, n_actions, n_states)
        self.transitions = self.transitions / self.transitions.sum(
            axis=2, keepdims=True
        )

        # Define rewards
        self.rewards = np.random.randn(n_states, n_actions)

    def reset(self) -> int:
        """Reset to initial state"""
        self.current_state = 0
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """Take action and return next state, reward, done"""
        if action >= self.n_actions:
            action = self.n_actions - 1

        # Sample next state
        next_state_probs = self.transitions[self.current_state, action]
        next_state = np.random.choice(self.n_states, p=next_state_probs)

        # Get reward
        reward = self.rewards[self.current_state, action]

        # Check if terminal
        done = next_state == self.n_states - 1
        truncated = False

        self.current_state = next_state

        return next_state, reward, done, truncated, {}


def create_environment(env_name: str = "CartPole-v1") -> EnvironmentWrapper:
    """Factory function to create environment wrapper

    Args:
        env_name: Name of the environment

    Returns:
        EnvironmentWrapper instance
    """
    return EnvironmentWrapper(env_name)


def get_environment_info(env: EnvironmentWrapper) -> dict:
    """Get information about the environment

    Args:
        env: Environment wrapper

    Returns:
        Dictionary with environment information
    """
    return {
        "name": env.env_name if env.env else "Mock",
        "state_size": env.state_size,
        "action_size": env.action_size,
        "is_continuous": env.is_continuous,
        "action_space_type": "continuous" if env.is_continuous else "discrete",
    }


def test_environment(env: EnvironmentWrapper, n_steps: int = 10) -> dict:
    """Test environment by taking random actions

    Args:
        env: Environment to test
        n_steps: Number of steps to test

    Returns:
        Dictionary with test results
    """
    state = env.reset()
    total_reward = 0
    states_visited = []
    actions_taken = []

    for _ in range(n_steps):
        states_visited.append(state.copy() if isinstance(state, np.ndarray) else state)

        if env.is_continuous:
            action = np.random.randn(env.action_size) * 0.1
        else:
            action = np.random.randint(env.action_size)

        actions_taken.append(action)
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state

        if done or truncated:
            break

    env.close()

    return {
        "total_reward": total_reward,
        "states_visited": len(states_visited),
        "actions_taken": actions_taken,
        "final_state": state,
    }
