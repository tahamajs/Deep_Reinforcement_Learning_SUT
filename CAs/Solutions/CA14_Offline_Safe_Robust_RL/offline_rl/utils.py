"""
Utilities for Offline Reinforcement Learning

This module provides utility functions for generating offline datasets.
"""

import numpy as np
from .dataset import OfflineDataset


def generate_offline_dataset(env_name="CartPole-v1", dataset_type="mixed", size=50000):
    """Generate offline dataset with different quality levels."""

    class SimpleGridWorld:
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

    env = SimpleGridWorld(size=5)

    states, actions, rewards, next_states, dones = [], [], [], [], []

    for _ in range(size):
        state = env.reset()
        episode_done = False
        episode_length = 0

        while not episode_done and episode_length < 50:
            if dataset_type == "expert":
                if state[0] < env.goal[0]:
                    action = 3  # right
                elif state[1] < env.goal[1]:
                    action = 0  # up
                else:
                    action = np.random.randint(4)
            elif dataset_type == "random":
                action = np.random.randint(4)
            else:  # mixed
                if np.random.random() < 0.7:
                    if state[0] < env.goal[0]:
                        action = 3  # right
                    elif state[1] < env.goal[1]:
                        action = 0  # up
                    else:
                        action = np.random.randint(4)
                else:
                    action = np.random.randint(4)

            next_state, reward, done, _ = env.step(action)

            states.append(state.copy())
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state.copy())
            dones.append(done)

            state = next_state
            episode_done = done
            episode_length += 1

            if episode_done:
                break

    return OfflineDataset(states, actions, rewards, next_states, dones, dataset_type)
