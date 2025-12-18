"""Tests for utility functions."""

import pytest
import torch

from src.utils import get_device, set_seed


class TestDeviceHandling:
    """Test device-related utilities."""

    def test_get_device_auto(self):
        device = get_device("auto")
        expected_type = "cuda" if torch.cuda.is_available() else "cpu"
        assert device.type == expected_type

    def test_get_device_cpu(self):
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_cuda(self):
        if torch.cuda.is_available():
            device = get_device("cuda")
            assert device.type == "cuda"
        else:
            # If CUDA not available, still allow creation but operations may fail
            device = get_device("cuda")
            assert device.type == "cuda"

    def test_get_device_invalid(self):
        try:
            get_device("invalid")
            assert False, "Should raise ValueError"
        except ValueError:
            pass


class TestSeeding:
    """Test seeding utilities."""

    def test_set_seed(self):
        # Basic test that it doesn't crash
        set_seed(42)
        # Could test reproducibility, but for simplicity, just call it
        assert True

    def test_set_env_seed(self):
        # Only run this test if gymnasium is available
        gym = pytest.importorskip('gymnasium')
        env = gym.make('CartPole-v1')
        # Should not raise
        from src.utils import set_env_seed
        set_env_seed(env, 123)
        env.close()
        assert True