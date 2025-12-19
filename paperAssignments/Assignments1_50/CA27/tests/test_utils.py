"""Unit tests for utility functions and trajectory collection."""
import unittest
import torch
from src.utils import Trajectory, collect_trajectory


class DummyEnvOldAPI:
    """Emulates older Gym API: reset->obs, step->(obs, reward, done, info)"""
    def __init__(self):
        self.observation_space = type('S', (), {'shape': (4,)})
        self.action_space = type('A', (), {'n': 2})
        self._state = [0.0, 0.0, 0.0, 0.0]

    def reset(self):
        return self._state

    def step(self, action):
        return self._state, 1.0, True, {}


class DummyEnvNewAPI:
    """Emulates newer Gym API: reset->(obs, info), step->(obs, reward, terminated, truncated, info)"""
    def __init__(self):
        self.observation_space = type('S', (), {'shape': (4,)})
        self.action_space = type('A', (), {'n': 2})
        self._state = [0.0, 0.0, 0.0, 0.0]

    def reset(self):
        return (self._state, {})

    def step(self, action):
        return (self._state, 1.0, False, True, {})


class SimplePolicy:
    """A deterministic simple policy returning logits for discrete actions."""
    def __call__(self, x):
        # Always prefer action 0
        return torch.tensor([[1.0, 0.0]])


class TestUtils(unittest.TestCase):
    def test_trajectory_add_and_to_tensors(self):
        traj = Trajectory()
        traj.add([0.0, 0.0, 0.0, 0.0], 0, 1.0)
        tensors = traj.to_tensors()
        self.assertIn('states', tensors)
        self.assertIn('rewards', tensors)

    def test_collect_trajectory_old_api(self):
        env = DummyEnvOldAPI()
        policy = SimplePolicy()
        traj = collect_trajectory(env, policy)
        self.assertGreaterEqual(len(traj), 1)

    def test_collect_trajectory_new_api(self):
        env = DummyEnvNewAPI()
        policy = SimplePolicy()
        traj = collect_trajectory(env, policy)
        self.assertGreaterEqual(len(traj), 1)


if __name__ == '__main__':
    unittest.main()
