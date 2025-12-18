"""Unit tests for meta-learning algorithms."""
import unittest
import torch
import numpy as np
from src.config import MAMLConfig, RL2Config
from src.maml import MAML
from src.rl2 import RL2Trainer
from src.tasks import CartPoleTask


class TestMAML(unittest.TestCase):
    """Test MAML implementation."""
    def setUp(self):
        self.config = MAMLConfig(obs_dim=4, action_dim=2)
        self.maml = MAML(self.config)

    def test_initialization(self):
        """Test MAML initialization."""
        self.assertIsNotNone(self.maml.policy)
        self.assertIsNotNone(self.maml.meta_optimizer)

    def test_policy_forward(self):
        """Test policy forward pass."""
        x = torch.randn(1, 4)
        output = self.maml.policy(x)
        self.assertEqual(output.shape, (1, 2))


class TestRL2(unittest.TestCase):
    """Test RL² implementation."""
    def setUp(self):
        self.config = RL2Config(obs_dim=4, action_dim=2)
        self.rl2 = RL2Trainer(self.config)

    def test_initialization(self):
        """Test RL² initialization."""
        self.assertIsNotNone(self.rl2.policy)
        self.assertIsNotNone(self.rl2.optimizer)

    def test_hidden_init(self):
        """Test hidden state initialization."""
        hidden = self.rl2.policy.init_hidden()
        self.assertEqual(len(hidden), 2)
        self.assertEqual(hidden[0].shape, (2, 1, 256))


class TestTasks(unittest.TestCase):
    """Test task implementations."""
    def test_cartpole_task(self):
        """Test CartPole task creation."""
        task = CartPoleTask()
        self.assertEqual(task.obs_dim, 4)
        self.assertEqual(task.action_dim, 2)
        self.assertTrue(task.is_discrete)


if __name__ == '__main__':
    unittest.main()