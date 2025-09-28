"""
Expert Data Collector for Imitation Learning

This module handles loading expert policies and collecting demonstration data.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import os
import pickle
import numpy as np
import gym
import tensorflow as tf
from tf_util import initialize


class ExpertDataCollector:
    """Class for collecting expert demonstration data."""

    def __init__(self, policy_fn, env_name, max_timesteps=None):
        """Initialize the data collector.

        Args:
            policy_fn: Function that takes observation and returns action
            env_name: Name of the gym environment
            max_timesteps: Maximum timesteps per episode (default: env limit)
        """
        self.policy_fn = policy_fn
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.max_timesteps = max_timesteps or self.env.spec.timestep_limit

    def collect_rollouts(self, num_rollouts, render=False):
        """Collect expert demonstration data.

        Args:
            num_rollouts: Number of expert rollouts to collect
            render: Whether to render the environment

        Returns:
            dict: Dictionary containing observations, actions, and returns
        """
        returns = []
        observations = []
        actions = []

        for i in range(num_rollouts):
            print(f"Collecting rollout {i + 1}/{num_rollouts}")
            obs = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = self.policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)

                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1

                if render:
                    self.env.render()

                if steps % 100 == 0:
                    print(f"{steps}/{self.max_timesteps}")

                if steps >= self.max_timesteps:
                    break

            returns.append(total_reward)
            print(f"Rollout {i + 1} completed with return: {total_reward:.2f}")

        print(f"Mean return: {np.mean(returns):.2f}")
        print(f"Std of return: {np.std(returns):.2f}")

        return {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "returns": returns,
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
        }

    def save_data(self, data, filename):
        """Save collected data to file.

        Args:
            data: Dictionary containing the collected data
            filename: Path to save the data
        """
        expert_data = {"observations": data["observations"], "actions": data["actions"]}

        with open(filename, "wb") as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

        print(f"Data saved to {filename}")


def load_expert_policy(policy_file):
    """Load expert policy from pickle file.

    Args:
        policy_file: Path to the expert policy file

    Returns:
        policy_fn: Function that takes observation and returns action
    """
    from load_policy import load_policy

    return load_policy(policy_file)
