from pathlib import Path

from src.config import load_config, ExperimentConfig


def test_load_default(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text("name: test\nmodel: {input_dim: 2, hidden_dims: [16], output_dim: 1}\ntrain: {seed: 1, batch_size: 8, lr: 0.01, epochs: 2}\n")
    cfg = load_config(p)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "test"
    assert cfg.model.input_dim == 2
    assert cfg.train.epochs == 2
