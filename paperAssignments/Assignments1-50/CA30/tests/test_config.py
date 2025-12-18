from ca30.config import ExperimentConfig
from pathlib import Path


def test_load_save(tmp_path: Path):
    cfg = ExperimentConfig(seed=7, input_dim=4)
    p = tmp_path / "cfg.yaml"
    cfg.save(p)
    loaded = ExperimentConfig.load(p)
    assert loaded.seed == 7
    assert loaded.input_dim == 4
