import pytest
pytest.importorskip("yaml")

from pathlib import Path
import tempfile

from src.config import ExperimentConfig


def test_from_yaml_and_validation(tmp_path: Path):
    cfg = ExperimentConfig()
    assert cfg.env_name == "CartPole-v1"

    # write a simple yaml
    p = tmp_path / "cfg.yaml"
    p.write_text("env_name: Pendulum-v1\nlearning_rate: 0.01\nhidden_sizes: [32, 32]\n")
    c = ExperimentConfig.from_yaml(p)
    assert c.env_name == "Pendulum-v1"
    assert c.learning_rate == pytest.approx(0.01)
    assert c.hidden_sizes == [32, 32]

    # invalid values should raise
    with pytest.raises(ValueError):
        ExperimentConfig(learning_rate=-1.0)
    with pytest.raises(ValueError):
        ExperimentConfig(gamma=2.0)
