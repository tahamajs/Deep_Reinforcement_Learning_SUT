from ca30.config import ExperimentConfig
from pathlib import Path


def test_load_save(tmp_path: Path):
    cfg = ExperimentConfig(seed=7, input_dim=4)
    p = tmp_path / "cfg.yaml"
    cfg.save(p)
    loaded = ExperimentConfig.load(p)
    assert loaded.seed == 7
    assert loaded.input_dim == 4


def test_from_yaml(tmp_path: Path):
    p = tmp_path / "ex.yaml"
    p.write_text("seed: 11\ninput_dim: 5\n")
    loaded = ExperimentConfig.load(p)
    assert loaded.seed == 11
    assert loaded.input_dim == 5
