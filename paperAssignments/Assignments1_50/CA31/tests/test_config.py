"""Tests for config module."""
import tempfile
import yaml
from src.config import load_config


def test_load_config():
    """Test loading configuration from YAML."""
    config_data = {
        "env_name": "CartPole-v1",
        "gamma": 0.99,
        "learning_rate": 0.001,
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    config = load_config(config_path)
    assert config["env_name"] == "CartPole-v1"
    assert config["gamma"] == 0.99
    assert config["learning_rate"] == 0.001