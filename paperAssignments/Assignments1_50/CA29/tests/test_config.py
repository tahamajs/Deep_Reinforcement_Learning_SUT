"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest

from src.config import SACConfig, load_config, save_config


class TestSACConfig:
    """Test SACConfig dataclass."""

    def test_default_values(self):
        config = SACConfig()
        assert config.env_name == "HalfCheetah-v4"
        assert config.gamma == 0.99
        assert config.alpha == 0.2
        assert config.lr_actor == 3e-4
        assert config.lr_critic == 3e-4
        assert config.buffer_size == 1_000_000
        assert config.batch_size == 256
        assert config.num_steps == 1_000_000
        assert config.eval_freq == 10_000
        assert config.seed == 42
        assert config.device == "auto"
        assert config.log_dir == "results/sac_experiment"

    def test_custom_values(self):
        config = SACConfig(env_name="Ant-v4", seed=999, gamma=0.95)
        assert config.env_name == "Ant-v4"
        assert config.seed == 999
        assert config.gamma == 0.95


class TestConfigIO:
    """Test loading and saving configs."""

    def test_load_save_config(self):
        config = SACConfig(seed=123, env_name="Pendulum-v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            save_config(config, str(path))

            loaded_config = load_config(str(path))
            assert loaded_config.seed == 123
            assert loaded_config.env_name == "Pendulum-v1"
            assert loaded_config.gamma == 0.99  # Default value

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")